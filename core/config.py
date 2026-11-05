import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Server configuration
PORT = int(os.getenv("PORT", 9985))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Oracle Database Credentials
uname = os.getenv("DB_USERNAME")
pwd = os.getenv("DB_PASSWORD")
schema = os.getenv("SCHEMA")
SCHEMA = schema
wltpwd = os.getenv("WALLET_PASSWORD")

BASE_DIR = Path(__file__).resolve().parent.parent
wallet_dir_name = os.getenv("WALLET_DIR", "Wallet_CLOUDEX")
wallet_path = str((BASE_DIR / wallet_dir_name).resolve())
cdir = wallet_path if os.path.isdir(wallet_path) else None
wltloc = wallet_path if os.path.isdir(wallet_path) else None
dsn = os.getenv("DB_DSN")

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", 2))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", 10))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", 30))

# OCI Configuration
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", "config")
OCI_CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
OCI_REGION = os.getenv("OCI_REGION")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# OCI Generative AI & Vector Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROJECT_ID = os.getenv("LLM_PROJECT_ID")
MODEL_NAME = os.getenv("MODEL_NAME", "openai.gpt-oss-120b")
ENDPOINT = os.getenv("ENDPOINT")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "cohere.embed-v4.0")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1024))

# Langfuse Observability
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PROJECT_FLAG = os.getenv("LANGFUSE_PROJECT_FLAG", "EnterpriseAI_RecSys")
