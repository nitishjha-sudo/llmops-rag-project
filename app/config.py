import os

class Config:
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    INDEX_NAME = os.getenv("RAG_INDEX_NAME", "corporate-rag-index")
    ENV_NAME = os.getenv("ENV_NAME", "dev-environment")
