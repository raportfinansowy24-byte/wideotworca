import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def run_ffmpeg(cmd, timeout=120):
    """Run FFmpeg command with proper error logging."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        if e.stderr:
            logger.error(e.stderr.decode(errors='ignore'))
        raise

def concatenate_videos(video_paths, output_path):
    """
    Łączy wiele plików mp4 w jeden film.
    """

    concat_file = "/tmp/concat.txt"

    with open(concat_file, "w") as f:
        for path in video_paths:
            # Get absolute path
            abs_path = os.path.abspath(path)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it with: apt-get install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")

    return output_path

def get_audio_duration(audio_file):
    """Pobranie czasu trwania audio za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {audio_file}: {e}")
        return 5.0

def get_video_duration(video_file):
    """Pobranie czasu trwania wideo za pomocą ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"⚠️ Nie udało się pobrać czasu trwania {video_file}: {e}")
        return 5.0

def ffmpeg_supports_subtitles():
    """Check if FFmpeg has libass support for subtitles filter."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Check if 'subtitles' filter is available
        return 'subtitles' in result.stdout
    except Exception as e:
        logger.warning(f"⚠️ Could not check FFmpeg filters: {e}")
        return False
