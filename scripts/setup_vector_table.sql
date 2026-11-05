-- ==============================================================================
-- Oracle Database 23ai AI Vector Search Setup Script
-- Author: Abdallah AlNsour (Oracle ACE Apprentice)
-- ==============================================================================

-- 1. Create Native Vector Knowledge Base Table
CREATE TABLE ENTERPRISE_DOCUMENTS_VECTOR (
    DOC_ID VARCHAR2(64) PRIMARY KEY,
    TITLE VARCHAR2(255) NOT NULL,
    CATEGORY VARCHAR2(100),
    CONTENT CLOB,
    DOC_EMBEDDING VECTOR(1024, FLOAT32),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Optional: Create In-Memory Neighbor Graph Vector Index (HNSW)
-- Enables ultra-fast approximate nearest neighbor (ANN) search over millions of vectors
CREATE VECTOR INDEX doc_vector_hnsw_idx 
ON ENTERPRISE_DOCUMENTS_VECTOR (DOC_EMBEDDING)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- 3. Query Sample: Semantic Search via In-Database Cosine Distance
-- Binds a 1024-dimension float array to :query_vec
SELECT DOC_ID, TITLE, CATEGORY, 
       VECTOR_DISTANCE(DOC_EMBEDDING, :query_vec, COSINE) AS distance
FROM ENTERPRISE_DOCUMENTS_VECTOR
ORDER BY distance ASC
FETCH FIRST 5 ROWS ONLY;
