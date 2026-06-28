"""
Quota Management Module (ZeroGPU + API Rate Limiting)

Features:
- Intelligent backoff for 429 errors (quota exhausted)
- Burst protection with exponential delays
- Request queue optimization
- Concurrent job limiting
"""

import os
import time
import logging
import threading
from typing import Callable, Any, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class QuotaManager:
    """Manages API quotas and rate limiting."""

    def __init__(
        self,
        max_concurrent_jobs: int = 1,  # CRITICAL: Set to 1 for ZeroGPU efficiency
        quota_backoff_multiplier: float = 2.0,
        quota_initial_wait: int = 60,
        quota_max_wait: int = 3600,
    ):
        """
        Initialize quota manager.

        Args:
            max_concurrent_jobs: Maximum concurrent GPU jobs (default: 1 for efficiency)
            quota_backoff_multiplier: Exponential backoff multiplier
            quota_initial_wait: Initial wait seconds for quota errors (429)
            quota_max_wait: Maximum wait seconds
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.quota_backoff_multiplier = quota_backoff_multiplier
        self.quota_initial_wait = quota_initial_wait
        self.quota_max_wait = quota_max_wait

        self.active_jobs = 0
        self.job_queue = deque()
        self.lock = threading.Lock()
        self.quota_hit_count = 0
        self.last_quota_hit_time = None

        logger.info(
            f"✅ Quota Manager initialized (max concurrent: {max_concurrent_jobs}, "
            f"backoff: {quota_backoff_multiplier}x)"
        )

    def can_submit_job(self) -> bool:
        """Check if new job can be submitted."""
        with self.lock:
            return self.active_jobs < self.max_concurrent_jobs

    def acquire_job_slot(self) -> bool:
        """Acquire a job execution slot."""
        with self.lock:
            if self.active_jobs < self.max_concurrent_jobs:
                self.active_jobs += 1
                logger.info(f"📊 Job slot acquired ({self.active_jobs}/{self.max_concurrent_jobs})")
                return True
            logger.warning(f"❌ No available job slots (queue: {len(self.job_queue)})")
            return False

    def release_job_slot(self):
        """Release a job execution slot."""
        with self.lock:
            if self.active_jobs > 0:
                self.active_jobs -= 1
                logger.info(f"✅ Job slot released ({self.active_jobs}/{self.max_concurrent_jobs})")

    def wait_for_quota_recovery(self, attempt: int = 0) -> int:
        """
        Calculate exponential backoff wait time for quota exhaustion (429).

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Wait time in seconds
        """
        # Exponential backoff: 60, 120, 240, 480, 3600
        wait_seconds = min(
            self.quota_initial_wait * (self.quota_backoff_multiplier ** attempt),
            self.quota_max_wait,
        )

        self.quota_hit_count += 1
        self.last_quota_hit_time = datetime.now()

        logger.warning(
            f"🚦 Quota exhausted (429) - Attempt {attempt + 1}. "
            f"Waiting {int(wait_seconds)}s before retry... "
            f"(Total quota hits: {self.quota_hit_count})"
        )

        # Sleep with progress logging
        for i in range(int(wait_seconds)):
            remaining = int(wait_seconds) - i
            if remaining % 10 == 0 or remaining <= 5:
                logger.info(f"⏳ Quota recovery: {remaining}s remaining...")
            time.sleep(1)

        return int(wait_seconds)

    def execute_with_quota_protection(
        self,
        operation_name: str,
        func: Callable,
        max_retries: int = 5,
        initial_delay: int = 2,
    ) -> Tuple[bool, Any]:
        """
        Execute function with automatic quota protection and backoff.

        Args:
            operation_name: Name of operation (for logging)
            func: Function to execute
            max_retries: Maximum retry attempts
            initial_delay: Initial retry delay (doubles on each retry)

        Returns:
            Tuple of (success: bool, result: Any)
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 {operation_name} - Attempt {attempt + 1}/{max_retries}")
                result = func()
                return True, result

            except Exception as e:
                error_text = str(e).lower()

                # Check for quota exhaustion (429, RESOURCE_EXHAUSTED)
                if "429" in error_text or "resource_exhausted" in error_text:
                    wait_time = self.wait_for_quota_recovery(attempt)
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {operation_name} failed after {max_retries} attempts (quota limit)")
                        return False, None

                # Check for non-retryable errors (4xx)
                non_retryable = [
                    "400", "invalid_argument", "401", "403", "404", "422",
                    "malformed", "invalid", "unauthorized", "forbidden"
                ]
                if any(marker in error_text for marker in non_retryable):
                    logger.error(f"❌ {operation_name} non-retryable error: {e}")
                    return False, None

                # Retryable errors (5xx, timeout, connection)
                if attempt < max_retries - 1:
                    retry_delay = initial_delay ** (attempt + 1)
                    logger.warning(
                        f"⚠️  {operation_name} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ {operation_name} failed after {max_retries} attempts")
                    return False, None

        return False, None

    def get_status(self) -> dict:
        """Get quota manager status."""
        with self.lock:
            return {
                "active_jobs": self.active_jobs,
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "queue_size": len(self.job_queue),
                "quota_hits": self.quota_hit_count,
                "last_quota_hit": self.last_quota_hit_time.isoformat() if self.last_quota_hit_time else None,
            }


# Global quota manager instance
_quota_manager: QuotaManager = None


def get_quota_manager() -> QuotaManager:
    """Get or create global quota manager instance."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager(
            max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "1")),
            quota_backoff_multiplier=float(os.getenv("QUOTA_BACKOFF_MULTIPLIER", "2.0")),
            quota_initial_wait=int(os.getenv("QUOTA_INITIAL_WAIT_SECONDS", "60")),
            quota_max_wait=int(os.getenv("QUOTA_MAX_WAIT_SECONDS", "3600")),
        )
    return _quota_manager
