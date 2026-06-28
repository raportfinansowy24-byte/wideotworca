"""
Batch Processing Module

Optimizations:
1. Accumulate render requests
2. Deduplicate identical topics before submission
3. Process in batches to maximize GPU utilization
4. Schedule jobs to avoid concurrent GPU waste
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a job in batch queue."""
    job_id: str
    topic: str
    hashtags: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    priority: int = 0  # Higher = process first
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None


class BatchProcessor:
    """Processes jobs in optimized batches to reduce costs."""

    def __init__(
        self,
        enabled: bool = True,
        batch_timeout_seconds: int = 300,
        batch_min_size: int = 3,
        dedup_similar_topics: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            enabled: Enable batch processing
            batch_timeout_seconds: Wait this long to accumulate jobs before processing
            batch_min_size: Minimum jobs in batch before forced processing
            dedup_similar_topics: Deduplicate similar topics
        """
        self.enabled = enabled
        self.batch_timeout_seconds = batch_timeout_seconds
        self.batch_min_size = batch_min_size
        self.dedup_similar_topics = dedup_similar_topics

        self.queue: Dict[str, BatchJob] = {}  # job_id -> BatchJob
        self.topic_map: Dict[str, List[str]] = defaultdict(list)  # topic -> [job_ids]
        self.processed_batches = 0
        self.lock = threading.Lock()
        self.batch_thread = None
        self.running = False

        logger.info(
            f"✅ Batch Processor initialized (enabled: {enabled}, "
            f"timeout: {batch_timeout_seconds}s, min_size: {batch_min_size})"
        )

    def add_job(
        self,
        job_id: str,
        topic: str,
        hashtags: List[str] = None,
        webhook_url: str = None,
        priority: int = 0,
    ) -> bool:
        """
        Add job to batch queue.

        Returns:
            True if added, False if duplicate
        """
        with self.lock:
            if job_id in self.queue:
                logger.warning(f"⚠️  Job {job_id} already in queue (duplicate)")
                return False

            job = BatchJob(
                job_id=job_id,
                topic=topic,
                hashtags=hashtags or [],
                webhook_url=webhook_url,
                priority=priority,
            )

            self.queue[job_id] = job
            self.topic_map[topic].append(job_id)

            logger.info(
                f"📥 Job added to batch queue: {job_id} (queue size: {len(self.queue)})"
            )

            # Check if batch is ready
            if self._should_process_batch():
                self._process_batch_internal()

            return True

    def get_batch(self, max_size: int = 10) -> List[BatchJob]:
        """
        Get next batch of jobs to process.

        Applies deduplication and priority sorting.
        """
        with self.lock:
            if not self.queue:
                return []

            # Sort by priority (descending), then by created_at (ascending)
            sorted_jobs = sorted(
                self.queue.values(),
                key=lambda j: (-j.priority, j.created_at),
            )

            batch = sorted_jobs[:max_size]

            # Deduplication: same topic = process once
            if self.dedup_similar_topics:
                seen_topics = set()
                deduped_batch = []
                for job in batch:
                    if job.topic not in seen_topics:
                        deduped_batch.append(job)
                        seen_topics.add(job.topic)
                    else:
                        logger.info(
                            f"🔄 Deduplicating job {job.job_id} (same topic as earlier job)"
                        )

                batch = deduped_batch

            # Remove from queue and mark as submitted
            for job in batch:
                del self.queue[job.job_id]
                if job.job_id in self.topic_map[job.topic]:
                    self.topic_map[job.topic].remove(job.job_id)
                job.submitted_at = time.time()

            self.processed_batches += 1
            logger.info(f"📦 Batch #{self.processed_batches} ready: {len(batch)} jobs")

            return batch

    def _should_process_batch(self) -> bool:
        """Check if batch should be processed now."""
        if len(self.queue) < self.batch_min_size:
            return False

        # Check if oldest job is approaching timeout
        if self.queue:
            oldest_job = min(self.queue.values(), key=lambda j: j.created_at)
            age = time.time() - oldest_job.created_at
            if age >= self.batch_timeout_seconds:
                logger.info(
                    f"⏰ Batch timeout reached ({int(age)}s >= {self.batch_timeout_seconds}s)"
                )
                return True

        return False

    def _process_batch_internal(self):
        """Internal batch processing trigger."""
        batch = self.get_batch()
        if batch:
            logger.info(f"🚀 Processing batch of {len(batch)} jobs")

    def get_queue_stats(self) -> dict:
        """Get batch queue statistics."""
        with self.lock:
            return {
                "queue_size": len(self.queue),
                "unique_topics": len(self.topic_map),
                "processed_batches": self.processed_batches,
                "oldest_job_age_seconds": (
                    time.time() - min(j.created_at for j in self.queue.values())
                    if self.queue
                    else 0
                ),
            }


# Global batch processor instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get or create global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(
            enabled=os.getenv("BATCH_PROCESSING_ENABLED", "true").lower() == "true",
            batch_timeout_seconds=int(os.getenv("BATCH_TIMEOUT_SECONDS", "300")),
            batch_min_size=int(os.getenv("BATCH_MIN_SIZE", "3")),
        )
    return _batch_processor
