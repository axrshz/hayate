from queue import Empty, Queue
from typing import List

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.request import Request


class Scheduler:
    """Move requests from a waiting queue into the active inference batch."""

    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE):
        self.pool: Queue = Queue()
        self.current_batch: List[Request] = []
        self.request_id = 0
        self.max_batch_size = max_batch_size

    def add(self, request: Request) -> None:
        """Add a prepared request to the waiting queue."""
        self.pool.put(request)

    def remove(self, request: Request) -> bool:
        """Remove one request from the active batch or waiting queue."""
        was_active = any(item is request for item in self.current_batch)
        if was_active:
            self.current_batch = [
                item for item in self.current_batch if item is not request
            ]

        # Queue has no public remove method. Its mutex protects the waiting list.
        with self.pool.mutex:
            queue_index = next(
                (
                    index
                    for index, queued_request in enumerate(self.pool.queue)
                    if queued_request is request
                ),
                None,
            )
            if queue_index is None:
                return was_active
            del self.pool.queue[queue_index]
            self.pool.unfinished_tasks -= 1
            if self.pool.unfinished_tasks == 0:
                self.pool.all_tasks_done.notify_all()
            self.pool.not_full.notify()
            return True

    def cancelled_waiting(self) -> List[Request]:
        """Return waiting requests that have a pending cancellation."""
        with self.pool.mutex:
            return [
                request
                for request in self.pool.queue
                if request.is_cancel_requested
            ]

    def tick(self) -> List[Request]:
        """Remove completed requests, then admit requests up to the batch limit."""
        # Keep unfinished requests in the batch so they can continue decoding.
        self.current_batch = [r for r in self.current_batch if not r.is_completed]
        # New requests join at each tick; this supports continuous batching.
        while len(self.current_batch) < self.max_batch_size:
            try:
                request = self.pool.get_nowait()
            except Empty:
                break
            self.pool.task_done()
            if not request.is_completed:
                self.current_batch.append(request)
        return self.current_batch
