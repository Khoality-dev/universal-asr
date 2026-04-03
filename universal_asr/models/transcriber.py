"""Transcriber — wraps faster-whisper for inference."""

import logging
import threading

import numpy as np
from faster_whisper import WhisperModel

from universal_asr import config
from universal_asr.models.manager import model_manager

logger = logging.getLogger(__name__)


class Transcriber:
    """Manages loaded WhisperModel instances and runs transcription."""

    def __init__(self):
        self._models: dict[str, WhisperModel] = {}
        self._lock = threading.Lock()

    def _get_model(self, model_name: str | None = None) -> WhisperModel:
        """Get or lazily load a WhisperModel."""
        name = model_name or config.DEFAULT_MODEL
        if name in self._models:
            return self._models[name]

        with self._lock:
            if name in self._models:
                return self._models[name]

            model_path = model_manager.resolve_model(name)
            logger.info(
                "Loading model: %s (device=%s, compute_type=%s)",
                model_path, config.DEVICE, config.COMPUTE_TYPE,
            )
            whisper_model = WhisperModel(
                model_path,
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
            )
            self._models[name] = whisper_model
            logger.info("Model loaded: %s", name)
            return whisper_model

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                logger.info("Unloaded model: %s", model_name)
                return True
        return False

    def transcribe(
        self,
        audio: np.ndarray,
        model_name: str | None = None,
        language: str | None = None,
        mode: str | None = None,
        initial_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe audio.

        Returns (text, detected_language).
        """
        model = self._get_model(model_name)
        kwargs: dict = {}
        if language:
            kwargs["language"] = language
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        if mode == "fast":
            kwargs["beam_size"] = 1
            kwargs["without_timestamps"] = True

        segments, info = model.transcribe(audio, **kwargs)
        text = "".join(segment.text for segment in segments).strip()
        return text, info.language

    def detect_language(
        self,
        audio: np.ndarray,
        model_name: str | None = None,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Detect language from audio.

        Returns (language, confidence, [(lang, prob), ...]).
        """
        model = self._get_model(model_name)
        # faster-whisper's detect_language expects features, use first 30s
        # Pad or trim to 30 seconds (480000 samples at 16kHz)
        target_len = 30 * 16000
        if len(audio) > target_len:
            audio = audio[:target_len]

        language, probs, *_ = model.detect_language(audio)
        # probs is a list of (lang, prob) sorted by probability
        confidence = next((p for l, p in probs if l == language), 0.0)
        return language, confidence, probs

    def loaded_models(self) -> list[str]:
        """Return names of currently loaded models."""
        return list(self._models.keys())


transcriber = Transcriber()
