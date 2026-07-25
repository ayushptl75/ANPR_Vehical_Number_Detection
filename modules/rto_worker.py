"""Background RTO lookup worker: simple in-process queue and job status tracking.

This is a lightweight implementation suitable for demo and local use. It keeps
job state in-memory and runs a background thread to process lookup jobs.
"""
from __future__ import annotations

import threading
import queue
import uuid
import time
from typing import Any

from modules.database_manager import DatabaseManager

db = DatabaseManager()

_job_queue: "queue.Queue[tuple[str,str]]" = queue.Queue()
_job_status: dict[str, dict[str, Any]] = {}


def _worker_loop() -> None:
    while True:
        try:
            job_id, plate = _job_queue.get()
            _job_status[job_id] = {"status": "running", "plate": plate, "created_at": time.time()}
            try:
                data = db.fetch_rto_data(plate)
                if data:
                    # insert as import record
                    import_id = db.add_rto_import(data)
                    _job_status[job_id]["status"] = "completed"
                    _job_status[job_id]["result"] = {"import_id": import_id}
                else:
                    _job_status[job_id]["status"] = "not_found"
            except Exception as exc:
                _job_status[job_id]["status"] = "error"
                _job_status[job_id]["error"] = str(exc)
            finally:
                _job_queue.task_done()
        except Exception:
            time.sleep(1)


# Start background worker thread
_thread = threading.Thread(target=_worker_loop, daemon=True)
_thread.start()


def enqueue_lookup(plate: str) -> str:
    job_id = str(uuid.uuid4())
    _job_status[job_id] = {"status": "queued", "plate": plate, "created_at": time.time()}
    _job_queue.put((job_id, plate))
    return job_id


def get_job_status(job_id: str) -> dict[str, Any] | None:
    return _job_status.get(job_id)
