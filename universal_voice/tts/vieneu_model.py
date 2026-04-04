"""VieNeu-TTS model implementation."""

import io
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from universal_voice import config
from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class VieNeuTTSModel(BaseTTSModel):
    """VieNeu-TTS model — Vietnamese TTS with instant voice cloning.

    Wraps the vieneu Python SDK. Model ID is 'vieneu:<mode>' where mode
    is the configured TTS_MODE (turbo, turbo_gpu, standard, fast).
    """

    SAMPLE_RATE = 24_000

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._mode = config.TTS_MODE

    @property
    def model_id(self) -> str:
        return f"vieneu:{self._mode}"

    def get_engine(self):
        """Get or lazily initialize the VieNeu TTS engine."""
        if self._engine is not None:
            return self._engine

        with self._lock:
            if self._engine is not None:
                return self._engine

            from vieneu import Vieneu

            logger.info("Initializing VieNeu TTS (mode=%s)", self._mode)
            self._engine = Vieneu(mode=self._mode)
            logger.info("VieNeu TTS initialized")
            return self._engine

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        engine = self.get_engine()
        infer_kwargs: dict = {"text": text}

        if ref_audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio_bytes)
                tmp_path = tmp.name
            try:
                infer_kwargs["ref_audio"] = tmp_path
                if ref_text:
                    infer_kwargs["ref_text"] = ref_text
                audio = engine.infer(**infer_kwargs)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        elif voice_id:
            voice = engine.get_preset_voice(voice_id)
            infer_kwargs["voice"] = voice
            audio = engine.infer(**infer_kwargs)
        else:
            audio = engine.infer(**infer_kwargs)

        return self._to_wav_bytes(audio)

    def list_voices(self) -> list[dict]:
        engine = self.get_engine()
        presets = engine.list_preset_voices()
        return [{"id": vid, "name": desc} for desc, vid in presets]

    def check_health(self) -> dict:
        try:
            self.get_engine()
            return {"ok": True, "message": "VieNeu TTS loaded"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def is_loaded(self) -> Optional[bool]:
        return self._engine is not None

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()
