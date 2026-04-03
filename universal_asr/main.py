"""FastAPI application for Universal ASR."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from universal_asr import config
from universal_asr.models.manager import model_manager
from universal_asr.routers import health, transcription

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load default model on startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if config.DEFAULT_MODEL:
        logger.info("Pre-loading default model: %s", config.DEFAULT_MODEL)
        try:
            model_manager.resolve_model(config.DEFAULT_MODEL)
            logger.info("Default model ready: %s", config.DEFAULT_MODEL)
        except Exception:
            logger.exception("Failed to pre-load default model")
    yield


app = FastAPI(title="Universal ASR", lifespan=lifespan)

app.include_router(health.router)
app.include_router(transcription.router)


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    import uvicorn
    uvicorn.run(
        "universal_asr.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
