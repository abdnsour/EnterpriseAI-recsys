import os
import json
import logging
from io import BytesIO
import uuid
import oci
from fastapi import HTTPException

from core.config import (
    COMPARTMENT_ID, BUCKET_NAME, OCI_CONFIG_FILE, OCI_CONFIG_PROFILE, OCI_REGION
)

logger = logging.getLogger(__name__)

class AIManager:
    """
    Manager for OCI AI Speech (Batch Transcription & Audio Synthesis)
    and OCI Object Storage.
    """
    def __init__(self, compartment_id: str = None, bucket_name: str = None):
        self.compartment_id = compartment_id or COMPARTMENT_ID
        self.bucket_name = bucket_name or BUCKET_NAME
        
        # Load OCI configuration safely
        if os.path.exists(OCI_CONFIG_FILE):
            self.config = oci.config.from_file(OCI_CONFIG_FILE, OCI_CONFIG_PROFILE)
        else:
            self.config = {}

        self.os_config = self.config.copy()
        if OCI_REGION:
            self.config['region'] = OCI_REGION
            self.os_config['region'] = OCI_REGION
        
        self._speech_client = None
        self._object_storage_client = None
        self._namespace = None
        
    @property
    def speech_client(self):
        if self._speech_client is None:
            self._speech_client = oci.ai_speech.AIServiceSpeechClient(self.config)
        return self._speech_client

    @property
    def object_storage_client(self):
        if self._object_storage_client is None:
            self._object_storage_client = oci.object_storage.ObjectStorageClient(self.os_config)
        return self._object_storage_client

    @property
    def namespace(self):
        if self._namespace is None:
            self._namespace = self.object_storage_client.get_namespace().data
        return self._namespace

    def create_transcription_job(self, file_name: str, display_name: str = "TranscriptionJob", description: str = "Transcription_Job"):
        """Creates a batch transcription job for an audio file located in OCI Object Storage."""
        try:
            logger.info(f"Creating transcription job for file: {file_name}")
            clean_description = "".join([c if c.isalnum() or c in ['-', '_'] else '_' for c in description])
            
            create_details = oci.ai_speech.models.CreateTranscriptionJobDetails(
                compartment_id=self.compartment_id,
                input_location=oci.ai_speech.models.ObjectListInlineInputLocation(
                    location_type="OBJECT_LIST_INLINE_INPUT_LOCATION",
                    object_locations=[
                        oci.ai_speech.models.ObjectLocation(
                            namespace_name=self.namespace,
                            bucket_name=self.bucket_name,
                            object_names=[file_name]
                        )
                    ]
                ),
                output_location=oci.ai_speech.models.OutputLocation(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    prefix="transcription_output/"
                ),
                display_name=display_name,
                description=clean_description,
                additional_transcription_formats=["SRT"],
                model_details=oci.ai_speech.models.TranscriptionModelDetails(
                    domain="GENERIC",
                    language_code="en-US"
                )
            )
            
            response = self.speech_client.create_transcription_job(create_details)
            logger.info(f"Transcription job created successfully. Job ID: {response.data.id}")
            return response.data
        except oci.exceptions.ServiceError as e:
            logger.error(f"OCI Service Error creating transcription job: {str(e)}", exc_info=True)
            raise HTTPException(status_code=e.status, detail=f"OCI Service Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating transcription job: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def get_transcription_job(self, job_id: str):
        """Retrieves the status and metadata of a transcription job."""
        try:
            response = self.speech_client.get_transcription_job(job_id)
            return response.data
        except oci.exceptions.ServiceError as e:
            logger.error(f"OCI Service Error getting transcription job: {str(e)}", exc_info=True)
            raise HTTPException(status_code=e.status, detail=f"OCI Service Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting transcription job: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def get_transcription_result(self, job_id: str):
        """Fetches the transcription result text directly from the JSON output if the job is SUCCEEDED."""
        try:
            job = self.get_transcription_job(job_id)
            if job.lifecycle_state != "SUCCEEDED":
                return {"status": job.lifecycle_state, "transcription": None}
                
            prefix = job.output_location.prefix
            list_response = self.object_storage_client.list_objects(self.namespace, self.bucket_name, prefix=prefix).data
            
            transcription_text = ""
            for obj in list_response.objects:
                if obj.name.endswith(".json"):
                    get_response = self.object_storage_client.get_object(self.namespace, self.bucket_name, obj.name)
                    file_content = get_response.data.content.decode("utf-8")
                    data = json.loads(file_content)
                    
                    if "transcriptions" in data:
                        for transcription in data["transcriptions"]:
                            transcription_text += transcription.get("transcription", "") + "\n"
                    else:
                        transcription_text = json.dumps(data)
                        
            return {"status": job.lifecycle_state, "transcription": transcription_text.strip()}
        except Exception as e:
            logger.error(f"Error getting transcription result: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def synthesize_speech(self, text: str, voice_id: str = "Bob", output_format: str = "MP3", object_name: str = None, folder: str = "TTS_output"):
        """Synthesizes speech from text and uploads the audio to OCI Object Storage."""
        try:
            logger.info(f"Synthesizing speech for text length: {len(text)}")
            
            model_details = oci.ai_speech.models.TtsOracleTts1StandardModelDetails(voice_id=voice_id)
            speech_settings = oci.ai_speech.models.TtsOracleSpeechSettings(text_type="TEXT", output_format=output_format)
            configuration = oci.ai_speech.models.TtsOracleConfiguration(model_details=model_details, speech_settings=speech_settings)
            
            request = oci.ai_speech.models.SynthesizeSpeechDetails(
                text=text,
                compartment_id=self.compartment_id,
                configuration=configuration
            )
            
            response = self.speech_client.synthesize_speech(request)
            
            audio_data = BytesIO()
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                audio_data.write(chunk)
            audio_data.seek(0)
            
            if object_name:
                clean_name = f"{folder}/{object_name.split('.')[0]}.{output_format.lower()}"
            else:
                clean_name = f"{folder}/tts_output_{uuid.uuid4().hex}.{output_format.lower()}"
                
            logger.info(f"Uploading synthesized speech to OCI Object Storage as {clean_name}")
            self.object_storage_client.put_object(self.namespace, self.bucket_name, clean_name, audio_data)
            logger.info(f"Successfully synthesized and uploaded: {clean_name}")
            
            return {"status": "SUCCEEDED", "object_name": clean_name, "bucket_name": self.bucket_name}
        except oci.exceptions.ServiceError as e:
            logger.error(f"OCI Service Error synthesizing speech: {str(e)}", exc_info=True)
            raise HTTPException(status_code=e.status, detail=f"OCI Service Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

ai_manager = AIManager()
