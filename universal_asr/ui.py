"""Gradio UI — model management, transcription test, language detection."""

import time

import gradio as gr
import numpy as np

from universal_asr import config
from universal_asr.models.manager import model_manager
from universal_asr.models.transcriber import transcriber


def _get_model_choices() -> list[str]:
    """Get list of available models for dropdowns."""
    choices = []
    # Always include default
    if config.DEFAULT_MODEL:
        choices.append(config.DEFAULT_MODEL)
    # Add cached models
    for m in model_manager.list_models():
        name = m["name"]
        if name not in choices:
            choices.append(name)
        if m["id"] not in choices and m["id"] != name:
            choices.append(m["id"])
    return choices or ["base"]


def _refresh_models_table() -> list[list]:
    """Get models as table rows."""
    models = model_manager.list_models()
    if not models:
        return [["No models cached", "", ""]]
    return [[m["name"], f"{m['size_mb']} MB", m["id"]] for m in models]


def _pull_model(model_name: str) -> tuple[str, list[list]]:
    """Pull/download a model."""
    if not model_name.strip():
        return "Please enter a model name.", _refresh_models_table()
    try:
        model_manager.resolve_model(model_name.strip())
        return f"Model '{model_name}' is ready.", _refresh_models_table()
    except Exception as e:
        return f"Error: {e}", _refresh_models_table()


def _delete_model(model_id: str) -> tuple[str, list[list]]:
    """Delete a cached model."""
    if not model_id.strip():
        return "Please enter a model ID to delete.", _refresh_models_table()
    transcriber.unload_model(model_id.strip())
    if model_manager.delete_model(model_id.strip()):
        return f"Deleted '{model_id}'.", _refresh_models_table()
    return f"Model '{model_id}' not found.", _refresh_models_table()


def _transcribe(audio, model_name: str, language: str | None) -> str:
    """Transcribe audio from Gradio component."""
    if audio is None:
        return "No audio provided."

    try:
        # Gradio audio component returns (sample_rate, numpy_array)
        sr, waveform = audio
        # Convert to float32
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
            # Normalize int types
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = waveform / 32768.0
        # Mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        # Resample to 16kHz if needed
        if sr != 16000:
            duration = len(waveform) / sr
            target_len = int(duration * 16000)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, target_len),
                np.arange(len(waveform)),
                waveform,
            ).astype(np.float32)

        model = model_name.strip() if model_name and model_name.strip() else None
        lang = language.strip() if language and language.strip() else None

        start = time.perf_counter()
        text, detected_lang = transcriber.transcribe(waveform, model_name=model, language=lang)
        elapsed = time.perf_counter() - start

        return f"**Text:** {text}\n\n**Language:** {detected_lang} | **Time:** {elapsed:.2f}s"
    except Exception as e:
        return f"Error: {e}"


def _detect_language(audio, model_name: str) -> str:
    """Detect language from audio."""
    if audio is None:
        return "No audio provided."

    try:
        sr, waveform = audio
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
            if waveform.max() > 1.0 or waveform.min() < -1.0:
                waveform = waveform / 32768.0
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            duration = len(waveform) / sr
            target_len = int(duration * 16000)
            waveform = np.interp(
                np.linspace(0, len(waveform) - 1, target_len),
                np.arange(len(waveform)),
                waveform,
            ).astype(np.float32)

        model = model_name.strip() if model_name and model_name.strip() else None

        start = time.perf_counter()
        language, confidence, probs = transcriber.detect_language(waveform, model_name=model)
        elapsed = time.perf_counter() - start

        lines = [f"**Detected:** {language} ({confidence:.1%}) | **Time:** {elapsed:.2f}s\n"]
        lines.append("| Language | Probability |")
        lines.append("|----------|-------------|")
        for lang, prob in probs[:10]:
            bar = "█" * int(prob * 30)
            lines.append(f"| {lang} | {prob:.2%} {bar} |")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Universal ASR", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Universal ASR")

        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Audio",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    with gr.Row():
                        model_select = gr.Dropdown(
                            label="Model",
                            choices=_get_model_choices(),
                            value=config.DEFAULT_MODEL,
                            allow_custom_value=True,
                        )
                        lang_input = gr.Textbox(
                            label="Language hint (optional)",
                            placeholder="e.g. en, vi, ja",
                            max_lines=1,
                        )
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    result_output = gr.Markdown(label="Result")

            transcribe_btn.click(
                fn=_transcribe,
                inputs=[audio_input, model_select, lang_input],
                outputs=result_output,
            )

        with gr.Tab("Language Detection"):
            with gr.Row():
                with gr.Column():
                    detect_audio = gr.Audio(
                        label="Audio",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    detect_model = gr.Dropdown(
                        label="Model",
                        choices=_get_model_choices(),
                        value=config.DEFAULT_MODEL,
                        allow_custom_value=True,
                    )
                    detect_btn = gr.Button("Detect Language", variant="primary")
                with gr.Column():
                    detect_output = gr.Markdown(label="Result")

            detect_btn.click(
                fn=_detect_language,
                inputs=[detect_audio, detect_model],
                outputs=detect_output,
            )

        with gr.Tab("Models"):
            models_table = gr.Dataframe(
                headers=["Name", "Size", "ID"],
                value=_refresh_models_table,
                interactive=False,
            )
            with gr.Row():
                pull_input = gr.Textbox(
                    label="Pull model",
                    placeholder="e.g. base, vinai/PhoWhisper-base",
                    max_lines=1,
                    scale=3,
                )
                pull_btn = gr.Button("Pull", variant="primary", scale=1)
            pull_status = gr.Textbox(label="Status", interactive=False)

            pull_btn.click(
                fn=_pull_model,
                inputs=pull_input,
                outputs=[pull_status, models_table],
            )

            with gr.Row():
                delete_input = gr.Textbox(
                    label="Delete model (enter ID)",
                    placeholder="Model ID from table above",
                    max_lines=1,
                    scale=3,
                )
                delete_btn = gr.Button("Delete", variant="stop", scale=1)
            delete_status = gr.Textbox(label="Status", interactive=False)

            delete_btn.click(
                fn=_delete_model,
                inputs=delete_input,
                outputs=[delete_status, models_table],
            )

            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=_refresh_models_table, outputs=models_table)

    return demo
