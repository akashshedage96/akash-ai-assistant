from sqlalchemy import create_engine

from chatrequest import ChatRequest
from config import SQL_DATABASE
from database_operations import create_database_engine
from state_store import IngestionStateStore

from structured_processor import StructuredFileProcessor
from llm_operations import generate_llm_resposne


from fastapi import FastAPI, Request

from ingestion_service import run_ingestion, status
import logging

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

@app.post("/ingest")
def ingest():
    run_ingestion()      
    return status        


@app.post("/chat")
def chat( chatrequest:ChatRequest):
    user_query = chatrequest.query
    if not user_query:
        return {"error": "Hello ! please tell me how can I help you ?"}
    else:
        response = generate_llm_resposne(user_query)
        return {"response": response}
    

















