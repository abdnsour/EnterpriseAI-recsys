"""
Standalone Demonstration: OCI Generative AI Dense Vectors + Oracle 23ai In-Database Similarity Search.
Author: Abdallah AlNsour (Oracle ACE Apprentice)
"""
import array
from components.llm_manager import llm_manager
from components.db_manager import db_manager

def run_demo():
    print("=== 1. Generating Dense Vector Embeddings via OCI Generative AI ===")
    sample_text = "Oracle Database 23ai natively combines relational tables with AI Vector Search."
    print(f"Input text: '{sample_text}'")
    
    embed_response = llm_manager.embed_text(sample_text)
    embedding = embed_response.embeddings[0]
    print(f"✅ Generated embedding vector with dimension: {len(embedding)}")

    print("\n=== 2. Preparing Sanitized Float Array for Oracle 23ai Binding ===")
    query_vector = array.array('f', embedding)
    print(f"✅ Bound float array created: {type(query_vector)} with {len(query_vector)} floats.")

    print("\n=== 3. Executing In-Database VECTOR_DISTANCE Query ===")
    sql_query = """
    SELECT DOC_ID, TITLE, CONTENT,
           VECTOR_DISTANCE(DOC_EMBEDDING, :1, COSINE) AS distance
    FROM ENTERPRISE_DOCUMENTS_VECTOR
    ORDER BY distance ASC
    FETCH FIRST 5 ROWS ONLY
    """
    res = db_manager.run_query_with_columns(sql_query, [query_vector])
    if res.get("is_data_found"):
        print(f"✅ Retrieved {len(res['data'])} nearest neighbors from Oracle 23ai:")
        for idx, row in enumerate(res["data"], 1):
            print(f"  {idx}. [{row.get('TITLE')}] | Cosine Distance: {row.get('DISTANCE')}")
    else:
        print(f"Database query status: {res.get('data')}")

if __name__ == "__main__":
    run_demo()
