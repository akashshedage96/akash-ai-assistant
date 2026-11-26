import logging
import threading
from datetime import datetime, timezone



from database_operations import create_database_engine
from vectore_store import get_vectorstore
from registry import build_registry
from state_store import IngestionStateStore
from storage import BlobStorageService
from ingestion_pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

_lock = threading.Lock()
status = {"running": False, "last_started": None, "last_finished": None, "last_result": None}


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_ingestion():
    
    if not _lock.acquire(blocking=False):
        logger.info("Ingestion already running, skipping")
        return
    try:
        status.update(running=True, last_started=_now(), last_result=None)
        engine = create_database_engine()
        processors = build_registry(get_vectorstore(), engine)
        blobs = BlobStorageService(supported_extensions=processors.keys())
        IngestionPipeline(blobs, IngestionStateStore(engine), processors).run()
        status["last_result"] = "success"
    except Exception:
        logger.exception("Ingestion failed")
        status["last_result"] = "failed"
    finally:
        status.update(running=False, last_finished=_now())
        _lock.release()