"""
LTX-Video Automation Script with CashMaker Integration

Workflow:
1. Generate LTX video clips (10s + 8s) via Hugging Face Gradio
2. Submit clips to CashMaker API for rendering with effects
3. Poll job status and retrieve final video URLs
4. Handle ZeroGPU queue with intelligent polling
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import time
from datetime import datetime

try:
    from gradio_client import Client, handle_file
except ImportError:
    print("❌ Required package missing. Install with:")
    print("   pip install gradio_client")
    sys.exit(1)

from api_client import CashMakerAPIClient, RenderJob, RenderStatus

# ============================================================================
# CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
HF_SPACE_ID = os.getenv("LTX_SPACE_ID", "Lightricks/LTX-2-3")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "10"))
QUEUE_CHECK_INTERVAL = int(os.getenv("QUEUE_CHECK_INTERVAL", "5"))
MAX_QUEUE_WAIT_SECONDS = int(os.getenv("MAX_QUEUE_WAIT_SECONDS", "600"))

# Clip configurations: (duration, name, frames @ 24fps)
CLIP_CONFIGS = [
    (10.0, "clip_10s", 240),
    (8.0, "clip_8s", 192),
]

# Optimal parameters for ZeroGPU performance
VIDEO_PARAMS = {
    "steps": int(os.getenv("VIDEO_STEPS", "20")),
    "height": int(os.getenv("VIDEO_HEIGHT", "480")),
    "width": int(os.getenv("VIDEO_WIDTH", "480")),
    "seed": 42,
    "randomize_seed": False,
    "enhance_prompt": False,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "ltx_automation.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Output directory ready: {OUTPUT_DIR.absolute()}")
    return OUTPUT_DIR


def validate_inputs(image_path: str, prompt: str) -> Tuple[bool, Optional[str]]:
    """Validate input image and prompt."""
    if not image_path:
        return False, "Image path cannot be empty"

    if isinstance(image_path, str) and not image_path.startswith("http"):
        if not Path(image_path).exists():
            return False, f"Image file not found: {image_path}"

    if not prompt or len(prompt.strip()) < 5:
        return False, "Prompt must be at least 5 characters"

    if len(prompt) > 500:
        return False, "Prompt exceeds 500 character limit"

    return True, None


def init_gradio_client(space_id: str) -> Client:
    """Initialize Gradio client with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"🔗 Connecting to {space_id}... (attempt {attempt + 1}/{MAX_RETRIES})")
            client = Client(space_id)
            logger.info(f"✅ Successfully connected to {space_id}")
            return client
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"❌ Failed to connect after {MAX_RETRIES} attempts")
                raise


def wait_for_job_completion(job, job_name: str, max_wait_seconds: int = MAX_QUEUE_WAIT_SECONDS):
    """
    Poll job status until completion, handling ZeroGPU queue.

    Args:
        job: Gradio job object
        job_name: Human-readable job name for logging
        max_wait_seconds: Maximum time to wait for job completion

    Returns:
        True if job completed successfully, False if timed out or errored
    """
    start_time = time.time()
    check_count = 0

    while time.time() - start_time < max_wait_seconds:
        try:
            status = job.status()
            check_count += 1

            logger.info(
                f"📊 {job_name} status: {status.code} "
                f"(checks: {check_count}, elapsed: {int(time.time() - start_time)}s)"
            )

            if status.code == "FINISHED":
                logger.info(f"✅ {job_name} completed successfully")
                return True

            elif status.code == "FAILED":
                logger.error(f"❌ {job_name} failed: {status.message}")
                return False

            elif status.code in ["QUEUED", "PROCESSING"]:
                logger.info(f"⏳ {job_name} is {status.code.lower()}...")
                time.sleep(QUEUE_CHECK_INTERVAL)

            else:
                logger.warning(f"⚠️  Unknown status code: {status.code}")
                time.sleep(QUEUE_CHECK_INTERVAL)

        except Exception as e:
            logger.warning(f"⚠️  Error checking job status: {e}")
            time.sleep(QUEUE_CHECK_INTERVAL)

    logger.error(f"❌ {job_name} timed out after {max_wait_seconds}s")
    return False


def generate_ltx_clip(
    client: Client,
    image_path: str,
    prompt: str,
    duration: float,
    output_filename: str,
) -> Tuple[bool, Optional[str]]:
    """
    Generate a single LTX video clip with ZeroGPU queue handling.

    Args:
        client: Initialized Gradio client
        image_path: Path or URL to input image
        prompt: Motion description prompt
        duration: Clip duration in seconds
        output_filename: Output filename (without extension)

    Returns:
        Tuple of (success: bool, file_path: Optional[str])
    """
    job_name = f"LTX Clip: {output_filename} ({int(duration)}s)"
    logger.info(f"🎬 Starting {job_name}...")

    try:
        # Prepare image for API
        if isinstance(image_path, str) and not image_path.startswith("http"):
            image_input = handle_file(image_path)
        else:
            image_input = image_path

        # Submit job to Gradio API
        logger.info(f"📤 Submitting {job_name} to Hugging Face...")
        job = client.submit(
            api_name="/generate_video",
            input_image=image_input,
            prompt=prompt,
            duration=duration,
            enhance_prompt=VIDEO_PARAMS["enhance_prompt"],
            seed=VIDEO_PARAMS["seed"],
            randomize_seed=VIDEO_PARAMS["randomize_seed"],
            height=VIDEO_PARAMS["height"],
            width=VIDEO_PARAMS["width"],
        )

        # Wait for job completion with polling
        if not wait_for_job_completion(job, job_name):
            logger.error(f"❌ {job_name} did not complete")
            return False, None

        # Retrieve result
        try:
            result = job.result()
            if not result or len(result) < 2:
                logger.error(f"❌ Invalid result from {job_name}")
                return False, None

            video_path = result[0]  # First element is the video file path

            if not video_path:
                logger.error(f"❌ No video path in result for {job_name}")
                return False, None

            # Move/save video to output directory
            output_path = OUTPUT_DIR / f"{output_filename}.mp4"
            if isinstance(video_path, str) and Path(video_path).exists():
                import shutil

                shutil.copy(video_path, output_path)
                logger.info(f"✅ {job_name} saved: {output_path}")
                return True, str(output_path)
            else:
                logger.warning(f"⚠️  Video file not found at {video_path}")
                return True, str(video_path)

        except Exception as e:
            logger.error(f"❌ Error retrieving {job_name} result: {e}")
            return False, None

    except Exception as e:
        logger.error(f"❌ Error generating {job_name}: {e}")
        return False, None


def submit_to_cashmaker(
    api_client: CashMakerAPIClient,
    topic: str,
    hashtags: list = None,
    webhook_url: str = None,
) -> Tuple[bool, Optional[RenderJob]]:
    """
    Submit generated clips to CashMaker API for rendering.

    Args:
        api_client: Initialized API client
        topic: Video topic
        hashtags: Optional hashtags
        webhook_url: Optional webhook URL for callbacks

    Returns:
        Tuple of (success: bool, job: Optional[RenderJob])
    """
    logger.info(f"📤 Submitting to CashMaker API: {topic}")

    success, job = api_client.submit_render(
        topic=topic,
        webhook_url=webhook_url,
        hashtags=hashtags,
    )

    if success:
        logger.info(f"✅ CashMaker job submitted: {job.job_id}")
        return True, job
    else:
        logger.error(f"❌ Failed to submit to CashMaker")
        return False, None


def wait_for_cashmaker_job(
    api_client: CashMakerAPIClient,
    job_id: str,
    max_wait_seconds: int = 600,
) -> Tuple[bool, Optional[RenderStatus]]:
    """
    Poll CashMaker job until completion.

    Args:
        api_client: Initialized API client
        job_id: Job ID from submission
        max_wait_seconds: Maximum wait time

    Returns:
        Tuple of (success: bool, status: Optional[RenderStatus])
    """
    logger.info(f"⏳ Waiting for CashMaker job {job_id}...")
    success, status = api_client.poll_until_complete(job_id, max_wait_seconds=max_wait_seconds)

    return success, status


def generate_full_sequence(
    image_path: str,
    prompt: str,
    topic: str = None,
    hashtags: list = None,
    webhook_url: str = None,
    submit_to_api: bool = True,
    wait_for_render: bool = True,
) -> Tuple[bool, dict]:
    """
    Complete workflow: Generate LTX clips → Submit to CashMaker → Wait for completion.

    Args:
        image_path: Path or URL to input image
        prompt: Motion description prompt for LTX
        topic: Video topic for CashMaker (if None, uses prompt)
        hashtags: Optional hashtags for CashMaker
        webhook_url: Optional webhook URL for callbacks
        submit_to_api: Whether to submit generated clips to CashMaker API
        wait_for_render: Whether to wait for CashMaker rendering

    Returns:
        Tuple of (success: bool, results: dict with all data)
    """
    logger.info("=" * 80)
    logger.info(f"🚀 Starting full automation sequence")
    logger.info(f"   Image: {image_path}")
    logger.info(f"   LTX Prompt: {prompt[:100]}...")
    if topic:
        logger.info(f"   CashMaker Topic: {topic}")
    logger.info("=" * 80)

    # Validate inputs
    is_valid, error_msg = validate_inputs(image_path, prompt)
    if not is_valid:
        logger.error(f"❌ Validation failed: {error_msg}")
        return False, {"error": error_msg}

    # Ensure output directory exists
    ensure_output_dir()

    # Initialize Gradio client for LTX
    try:
        gradio_client = init_gradio_client(HF_SPACE_ID)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gradio client: {e}")
        return False, {"error": str(e)}

    # Initialize API client for CashMaker (optional)
    api_client = None
    if submit_to_api:
        api_client = CashMakerAPIClient()
        if not api_client.health_check():
            logger.warning("⚠️  CashMaker API not responding, skipping submission")
            api_client = None

    # Generate LTX clips
    results = {
        "timestamp": datetime.now().isoformat(),
        "image": image_path,
        "ltx_prompt": prompt,
        "ltx_clips": {},
        "cashmaker_job": None,
    }

    for duration, clip_name, num_frames in CLIP_CONFIGS:
        logger.info(f"\n📽️  Generating {clip_name} ({duration}s, {num_frames} frames @ 24fps)...")

        success, file_path = generate_ltx_clip(
            client=gradio_client,
            image_path=image_path,
            prompt=prompt,
            duration=duration,
            output_filename=clip_name,
        )

        if success:
            results["ltx_clips"][clip_name] = {
                "status": "success",
                "path": file_path,
                "duration": duration,
                "frames": num_frames,
            }
        else:
            results["ltx_clips"][clip_name] = {
                "status": "failed",
                "path": None,
                "error": f"Failed to generate {clip_name}",
            }

        # Small delay between sequential requests
        if duration == CLIP_CONFIGS[0][0]:
            logger.info("⏱️  Waiting before next clip generation...")
            time.sleep(5)

    # Check if all clips generated successfully
    all_clips_success = all(c.get("status") == "success" for c in results["ltx_clips"].values())

    if not all_clips_success:
        logger.warning("⚠️  Some LTX clips failed, skipping CashMaker submission")
        return False, results

    # Submit to CashMaker (optional)
    if api_client and submit_to_api:
        cashmaker_topic = topic or prompt[:100]
        success, job = submit_to_cashmaker(
            api_client=api_client,
            topic=cashmaker_topic,
            hashtags=hashtags,
            webhook_url=webhook_url,
        )

        if success:
            results["cashmaker_job"] = {
                "job_id": job.job_id,
                "status": job.status,
                "status_url": job.status_url,
            }

            # Wait for rendering (optional)
            if wait_for_render:
                success, status = wait_for_cashmaker_job(api_client, job.job_id)

                if success and status:
                    results["cashmaker_job"].update(
                        {
                            "final_status": status.state,
                            "video_url": status.video_url,
                            "video_duration": status.video_duration,
                            "file_size_mb": status.file_size_mb,
                        }
                    )
                else:
                    results["cashmaker_job"]["final_status"] = "failed"

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ Full automation sequence completed!")
    logger.info("\nLTX Clips:")
    for clip_name, clip_info in results["ltx_clips"].items():
        if clip_info.get("status") == "success":
            logger.info(f"   ✅ {clip_name}: {clip_info['path']}")
        else:
            logger.info(f"   ❌ {clip_name}: {clip_info.get('error')}")

    if results.get("cashmaker_job"):
        cj = results["cashmaker_job"]
        logger.info(f"\nCashMaker Job:")
        logger.info(f"   Job ID: {cj.get('job_id')}")
        logger.info(f"   Status: {cj.get('final_status', 'pending')}")
        if cj.get("video_url"):
            logger.info(f"   Final Video: {cj['video_url']}")

    logger.info("=" * 80)

    return all_clips_success, results


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LTX-Video + CashMaker Automation: Full workflow from image to rendered video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate LTX clips only
  python ltx_video_automation.py --image img.jpg --prompt "smooth pan" --no-cashmaker

  # Full workflow with CashMaker
  python ltx_video_automation.py --image img.jpg --prompt "zoom effect" --topic "My Video" --hashtags "#finance" "#stocks"

  # With webhook
  python ltx_video_automation.py --image img.jpg --prompt "motion" --webhook-url https://example.com/webhook
        """,
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path or URL to input image",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Motion description prompt for LTX (5-500 chars)",
    )
    parser.add_argument(
        "--topic",
        help="Video topic for CashMaker (optional, defaults to prompt)",
    )
    parser.add_argument(
        "--hashtags",
        nargs="+",
        help="Hashtags for CashMaker (e.g., --hashtags #finance #stocks)",
    )
    parser.add_argument(
        "--webhook-url",
        help="Webhook URL for CashMaker callbacks",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--no-cashmaker",
        action="store_true",
        help="Generate LTX clips only, don't submit to CashMaker",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit to CashMaker but don't wait for completion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Override globals from CLI args
    global OUTPUT_DIR, LOG_LEVEL
    OUTPUT_DIR = Path(args.output_dir)
    LOG_LEVEL = args.log_level

    # Reconfigure logging
    logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

    # Run automation
    success, results = generate_full_sequence(
        image_path=args.image,
        prompt=args.prompt,
        topic=args.topic,
        hashtags=args.hashtags,
        webhook_url=args.webhook_url,
        submit_to_api=not args.no_cashmaker,
        wait_for_render=not args.no_wait,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
