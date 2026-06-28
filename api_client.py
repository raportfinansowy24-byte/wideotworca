"""
CashMaker VEO Worker API Client

Provides programmatic access to the video rendering API with:
- Authentication via API key
- Job submission and polling
- Webhook integration
- Error handling and retry logic
"""

import os
import logging
import time
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RenderJob:
    """Represents a render job response."""

    job_id: str
    status: str
    status_url: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_response(cls, data: dict) -> "RenderJob":
        """Create RenderJob from API response."""
        return cls(
            job_id=data.get("job_id"),
            status=data.get("status"),
            status_url=data.get("status_url"),
            video_url=data.get("video_url"),
            error=data.get("error"),
            created_at=datetime.now().isoformat(),
        )


@dataclass
class RenderStatus:
    """Represents render status at a point in time."""

    job_id: str
    state: str
    status: str
    topic: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    video_duration: Optional[float] = None
    file_size_mb: Optional[float] = None

    @classmethod
    def from_response(cls, data: dict) -> "RenderStatus":
        """Create RenderStatus from API response."""
        return cls(
            job_id=data.get("job_id"),
            state=data.get("state"),
            status=data.get("status"),
            topic=data.get("topic"),
            video_url=data.get("video_url"),
            error=data.get("error"),
            video_duration=data.get("video_duration"),
            file_size_mb=data.get("file_size_mb"),
        )


class CashMakerAPIClient:
    """Client for CashMaker VEO Worker API."""

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize API client.

        Args:
            api_url: Base API URL (default: from WORKER_API_URL env var)
            api_key: API key for authentication (default: from WORKER_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_url = (api_url or os.getenv("WORKER_API_URL", "http://localhost:5000")).rstrip("/")
        self.api_key = api_key or os.getenv("WORKER_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            logger.warning("⚠️  No API key configured. Set WORKER_API_KEY environment variable.")

        self._session = requests.Session()
        self._session.headers.update(self._get_headers())

        logger.info(f"🔗 CashMaker API Client initialized: {self.api_url}")

    def _get_headers(self) -> dict:
        """Get standard request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict = None,
        params: dict = None,
    ) -> Tuple[bool, dict]:
        """
        Make HTTP request with retry logic.

        Returns:
            Tuple of (success: bool, data: dict)
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"📤 {method} {url}")

                response = self._session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=self.timeout,
                )

                response.raise_for_status()
                data = response.json()

                logger.debug(f"✅ Response: {data}")
                return True, data

            except requests.exceptions.RequestException as e:
                attempt_num = attempt + 1
                logger.warning(
                    f"⚠️  Request failed (attempt {attempt_num}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"❌ Request failed after {self.max_retries} attempts")
                    return False, {"error": str(e)}

        return False, {"error": "Request failed"}

    def submit_render(
        self,
        topic: str,
        webhook_url: str = None,
        hashtags: list = None,
        narration: dict = None,
    ) -> Tuple[bool, Optional[RenderJob]]:
        """
        Submit a render job.

        Args:
            topic: Video topic/prompt
            webhook_url: Optional webhook URL for callbacks
            hashtags: Optional list of hashtags (e.g., ["#finance", "#stocks"])
            narration: Optional narration dict with hook/problem/solution keys

        Returns:
            Tuple of (success: bool, job: Optional[RenderJob])
        """
        logger.info(f"🎬 Submitting render job: {topic[:50]}...")

        payload = {"topic": topic}
        if webhook_url:
            payload["webhookUrl"] = webhook_url
        if hashtags:
            payload["hashtags"] = hashtags
        if narration:
            payload["narration"] = narration

        success, data = self._request("POST", "/render-sequence", json_data=payload)

        if success and "job_id" in data:
            job = RenderJob.from_response(data)
            logger.info(f"✅ Job submitted: {job.job_id}")
            return True, job
        else:
            logger.error(f"❌ Failed to submit job: {data.get('error', 'Unknown error')}")
            return False, None

    def get_status(self, job_id: str) -> Tuple[bool, Optional[RenderStatus]]:
        """
        Get render job status.

        Args:
            job_id: Job ID from submit_render

        Returns:
            Tuple of (success: bool, status: Optional[RenderStatus])
        """
        logger.debug(f"📊 Checking status for job: {job_id}")

        success, data = self._request("GET", f"/status/{job_id}")

        if success:
            status = RenderStatus.from_response(data)
            logger.debug(f"   Status: {status.state}")
            return True, status
        else:
            logger.warning(f"⚠️  Failed to get status: {data.get('error', 'Unknown error')}")
            return False, None

    def poll_until_complete(
        self,
        job_id: str,
        max_wait_seconds: int = 600,
        poll_interval: int = 10,
    ) -> Tuple[bool, Optional[RenderStatus]]:
        """
        Poll job status until completion or timeout.

        Args:
            job_id: Job ID
            max_wait_seconds: Maximum time to wait
            poll_interval: Seconds between polls

        Returns:
            Tuple of (success: bool, final_status: Optional[RenderStatus])
        """
        logger.info(f"⏳ Polling job {job_id}... (max wait: {max_wait_seconds}s)")

        start_time = time.time()
        poll_count = 0

        while time.time() - start_time < max_wait_seconds:
            success, status = self.get_status(job_id)
            poll_count += 1

            if not success:
                logger.warning(f"⚠️  Failed to get status (poll #{poll_count})")
                time.sleep(poll_interval)
                continue

            elapsed = int(time.time() - start_time)
            logger.info(
                f"📊 Poll #{poll_count} | Status: {status.state} | Elapsed: {elapsed}s"
            )

            if status.state in ["success", "completed"]:
                logger.info(f"✅ Job completed! Video URL: {status.video_url}")
                return True, status

            elif status.state == "failed":
                logger.error(f"❌ Job failed: {status.error}")
                return False, status

            time.sleep(poll_interval)

        logger.error(f"❌ Job timed out after {max_wait_seconds}s")
        return False, status

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            success, data = self._request("GET", "/health")
            if success:
                logger.info(f"✅ API health check passed")
                return True
            else:
                logger.warning(f"⚠️  API health check failed: {data}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False


# ============================================================================
# Convenience Functions
# ============================================================================


def create_client(
    api_url: str = None,
    api_key: str = None,
) -> CashMakerAPIClient:
    """Create a default API client."""
    return CashMakerAPIClient(api_url=api_url, api_key=api_key)


def submit_and_wait(
    topic: str,
    webhook_url: str = None,
    hashtags: list = None,
    max_wait_seconds: int = 600,
) -> Tuple[bool, Optional[RenderStatus]]:
    """
    Convenience function to submit job and wait for completion.

    Args:
        topic: Video topic
        webhook_url: Optional webhook URL
        hashtags: Optional hashtags
        max_wait_seconds: Max wait time

    Returns:
        Tuple of (success: bool, final_status: Optional[RenderStatus])
    """
    client = create_client()

    # Check health
    if not client.health_check():
        logger.error("❌ API not responding")
        return False, None

    # Submit job
    success, job = client.submit_render(
        topic=topic,
        webhook_url=webhook_url,
        hashtags=hashtags,
    )

    if not success:
        return False, None

    # Poll until complete
    return client.poll_until_complete(job.job_id, max_wait_seconds=max_wait_seconds)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    client = create_client()

    if client.health_check():
        success, job = client.submit_render(
            topic="Inwestowanie w akcje - 5 sposobów na zarobek",
            webhook_url="https://example.com/webhook",
            hashtags=["#finanse", "#akcje"],
        )

        if success:
            print(f"Job submitted: {job.job_id}")
            print(f"Status URL: {job.status_url}")
