"""
Piper TTS integration for Polish voice synthesis.

Voice model: pl_PL-mc_speech-medium
"""

import os
import logging

logger = logging.getLogger(__name__)

PIPER_VOICE = "pl_PL-mc_speech-medium"
PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")


def ensure_voice_downloaded():
    """Check if the Polish Piper voice model is cached; download it if not.

    Uses piper.download_voices() to fetch and cache the model in the default
    Piper voice directory (~/.local/share/piper/voices/).

    Returns True if the voice is available after the call, False on failure.
    """
    try:
        import piper

        os.makedirs(PIPER_VOICE_DIR, exist_ok=True)

        # Check whether the model files are already present.  Piper stores the
        # ONNX model as "<voice>.onnx" and its config as "<voice>.onnx.json".
        model_file = os.path.join(PIPER_VOICE_DIR, f"{PIPER_VOICE}.onnx")
        config_file = os.path.join(PIPER_VOICE_DIR, f"{PIPER_VOICE}.onnx.json")

        if os.path.exists(model_file) and os.path.exists(config_file):
            logger.info(f"✅ Piper voice already cached: {PIPER_VOICE}")
            return True

        logger.info(f"⬇️  Downloading Piper voice model: {PIPER_VOICE} ...")
        piper.download_voices([PIPER_VOICE], download_dir=PIPER_VOICE_DIR)
        logger.info(f"✅ Piper voice downloaded and cached: {PIPER_VOICE}")
        return True

    except Exception as e:
        logger.warning(
            f"⚠️  Failed to download Piper voice '{PIPER_VOICE}': {e}. "
            "Voice will be downloaded on first TTS call."
        )
        return False


def generate_piper_tts(text: str, output_path: str) -> str:
    """Synthesise *text* to a WAV file at *output_path* using Piper TTS.

    Ensures the voice model is present before synthesis.  Raises RuntimeError
    if synthesis fails.

    Args:
        text: Polish text to synthesise.
        output_path: Destination file path (should end in .wav).

    Returns:
        The resolved *output_path* on success.
    """
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise RuntimeError(
            "piper-tts is not installed. Add 'piper-tts' to requirements.txt."
        ) from exc

    # Make sure the model is available before we try to load it.
    ensure_voice_downloaded()

    model_file = os.path.join(PIPER_VOICE_DIR, f"{PIPER_VOICE}.onnx")
    config_file = os.path.join(PIPER_VOICE_DIR, f"{PIPER_VOICE}.onnx.json")

    if not os.path.exists(model_file):
        raise RuntimeError(
            f"Piper voice model not found at {model_file}. "
            "Download failed during startup and on-demand download also failed."
        )

    try:
        logger.info(f"🔊 Piper TTS: synthesising {len(text)} chars → {output_path}")
        voice = PiperVoice.load(model_file, config_path=config_file)
        with open(output_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        logger.info(f"✅ Piper TTS: audio saved to {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Piper TTS synthesis failed: {e}") from e


# ---------------------------------------------------------------------------
# Module-level: attempt to pre-download the voice when the module is imported
# so that the first TTS call is not delayed by a network download.
# ---------------------------------------------------------------------------
ensure_voice_downloaded()

