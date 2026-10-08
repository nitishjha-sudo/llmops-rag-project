import os
import logging
from fastapi import FastAPI, HTTPException
from app.rag_pipeline import get_rag_chain
from app.database import seed_initial_data
from langsmith import Client

app = FastAPI(title="Corporate LLMOps RAG API")
langsmith_client = Client()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s] - %(message)s')
logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/seed")
async def seed_db():
    try:
        return seed_initial_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(user_input: str):
    chain = get_rag_chain()
    response = chain.invoke({"input": user_input})
    logger.info(f"RAG query executed", extra={"metric_name": "rag.queries", "metric_value": 1})
    return {
        "answer": response["answer"],
        "run_id": response.get("run_id") 
    }

@app.post("/feedback")
async def log_feedback(run_id: str, rating: int, comment: str = None):
    langsmith_client.create_feedback(run_id=run_id, key="user-score", score=rating, comment=comment)
    return {"status": "Feedback logged successfully"}