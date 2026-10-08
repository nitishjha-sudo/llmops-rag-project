import os
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.config import Config

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = Config.ENV_NAME

def get_rag_chain():
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=Config.AWS_REGION)
    
    # Fully updated to use Amazon Nova Lite with standard cross-region endpoint syntax
    llm = ChatBedrock(model_id=f"us.{os.getenv('AMAZON_NOVA_ID', 'amazon.nova-lite-v1:0')}", region_name=Config.AWS_REGION)
    
    vector_store = OpenSearchVectorSearch(
        index_name=Config.INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=Config.OPENSEARCH_URL
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based strictly on the provided context:
    Context: {context}
    Question: {input}
    Answer:""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)
