"""
Parler TTS integration for Polish voice synthesis.

Uses HuggingFace Inference API with Parler TTS Mini model.
Voice: Polish Alex (speaker_id: 25849)
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

PARLER_MODEL = "parler-tts/parler-tts-mini"
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_piper_tts(text: str, output_path: str) -> str:
    """Synthesise *text* to a WAV file at *output_path* using Parler TTS.

    Uses HuggingFace Inference API with Polish Alex voice.

    Args:
        text: Polish text to synthesise.
        output_path: Destination file path (should end in .wav).

    Returns:
        The resolved *output_path* on success.
    
    Raises:
        RuntimeError: If synthesis fails.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=HF_TOKEN)
        
        logger.info(f"🔊 Parler TTS: synthesising {len(text)} chars → {output_path}")
        
        # Parler TTS prompt format: "Speaker description: text"
        # Polish Alex voice (speaker_id: 25849)
        prompt = f"A male speaker with a clear, professional voice: {text}"
        
        # Call Parler TTS via HuggingFace Inference
        audio = client.text_to_speech(
            text=prompt,
            model=PARLER_MODEL,
        )
        
        # Save audio to file
        with open(output_path, "wb") as f:
            f.write(audio)
        
        logger.info(f"✅ Parler TTS: audio saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Parler TTS synthesis failed: {e}")
        raise RuntimeError(f"Parler TTS synthesis failed: {e}") from e

