"""
Fixed Wan2.1 Gradio API integration that properly handles async responses.
"""

import os
import time
import logging
import requests
import tempfile
import shutil
from gradio_client import Client

logger = logging.getLogger(__name__)

def generate_hunyuan_video_segment(prompt, output_path, aspect_ratio="9:16"):
    """Generate video via Wan2.1 Gradio API with proper async handling."""
    
    logger.info(f"🎬 Wan2.1: generating video for prompt: {prompt[:60]}...")
    
    try:
        client = Client("Wan-AI/Wan2.1")
        logger.info(f"✅ Connected to Wan2.1 Gradio Space")
        
        # Map aspect ratio to size
        size = '1280*720'
        if aspect_ratio == "9:16":
            size = '720*1280'
        elif aspect_ratio == "1:1":
            size = '960*960'
        
        logger.info(f"⏳ Submitting job to Wan2.1 with size {size}...")
        
        # Submit async job
        job = client.submit(
            prompt=prompt,
            size=size,
            watermark_wan=True,
            seed=-1,
            api_name="/t2v_generation_async"
        )
        
        logger.info("✅ Job submitted, waiting for completion...")
        
        # Wait for job to complete
        result = job.result()
        
        logger.info(f"✅ Job completed. Result type: {type(result)}")
        logger.info(f"Result: {result}")
        
        # Parse result - Gradio returns tuple of outputs
        video_path = None
        
        if isinstance(result, (list, tuple)):
            # Result is typically (video_file, metadata_dict)
            for item in result:
                if isinstance(item, str):
                    # Check if it's a file path or URL
                    if item.endswith('.mp4') or item.startswith('http') or item.startswith('/'):
                        video_path = item
                        logger.info(f"✅ Found video path: {video_path}")
                        break
                elif isinstance(item, dict):
                    # Check for 'name' or 'path' key
                    if 'name' in item:
                        video_path = item['name']
                        logger.info(f"✅ Found video in dict['name']: {video_path}")
                        break
                    elif 'path' in item:
                        video_path = item['path']
                        logger.info(f"✅ Found video in dict['path']: {video_path}")
                        break
        elif isinstance(result, dict):
            if 'name' in result:
                video_path = result['name']
            elif 'path' in result:
                video_path = result['path']
        elif isinstance(result, str):
            video_path = result
        
        if not video_path:
            logger.error(f"❌ Could not extract video path from result: {result}")
            raise RuntimeError(f"No video path found in Wan2.1 response: {result}")
        
        # Download video if it's a URL
        if video_path.startswith('http'):
            logger.info(f"💾 Downloading video from {video_path[:80]}...")
            response = requests.get(video_path, timeout=300)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✅ Video saved to {output_path}")
        else:
            # It's a local file path - copy it
            logger.info(f"📋 Copying video from {video_path} to {output_path}...")
            shutil.copy(video_path, output_path)
            logger.info(f"✅ Video copied to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Wan2.1 generation failed: {e}")
        raise RuntimeError(f"Wan2.1 video generation failed: {e}") from e

