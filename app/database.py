import logging
from opensearchpy import OpenSearch
from app.config import Config
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings

logger = logging.getLogger(__name__)

def seed_initial_data():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0", 
        region_name=Config.AWS_REGION
    )
    
    sample_texts = [
        "The corporate office hybrid policy requires all employees to work from the office on Tuesdays and Thursdays.",
        "For expense reimbursement, all receipts above $25 must be submitted via the internal portal within 30 days.",
        "The standard internal IT support channel on Slack is #help-it-tickets."
    ]
    
    logger.info("Connecting to OpenSearch to seed documents...")
    vector_store = OpenSearchVectorSearch.from_texts(
        texts=sample_texts,
        embedding=embeddings,
        opensearch_url=Config.OPENSEARCH_URL,
        index_name=Config.INDEX_NAME
    )
    logger.info("Successfully seeded OpenSearch vector database!")
    return {"status": "Database successfully seeded"}
