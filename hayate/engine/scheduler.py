from queue import Queue
from typing import List

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.request import Request


class Scheduler:
    """Manages the request queue and active batch for continuous batching."""

    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE):
        self.pool: Queue = Queue()
        self.current_batch: List[Request] = []
        self.request_id = 0
        self.max_batch_size = max_batch_size

    def add(self, request: Request) -> None:
        """Queue a prepared request."""
        self.pool.put(request)

    def tick(self) -> List[Request]:
        """Admit queued requests and evict completed ones. Returns the active batch."""
        self.current_batch = [r for r in self.current_batch if not r.is_completed]
        while not self.pool.empty() and len(self.current_batch) < self.max_batch_size:
            self.current_batch.append(self.pool.get())
        return self.current_batch
