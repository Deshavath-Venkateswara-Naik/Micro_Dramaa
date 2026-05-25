from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import ingestion, segmentation, audio, speech, bgm, fusion, sequencer, renderer, story
import uvicorn
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Micro-Drama AI Pipeline", description="Backend for Micro-Drama generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose local storage directory so the external Subtitle API can download the audio files
storage_path = os.path.join(os.path.dirname(__file__), "..", "storage")
app.mount("/storage", StaticFiles(directory=storage_path), name="storage")

app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(segmentation.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
app.include_router(speech.router, prefix="/api/v1")
app.include_router(bgm.router, prefix="/api/v1")
app.include_router(fusion.router, prefix="/api/v1")
app.include_router(sequencer.router, prefix="/api/v1")
app.include_router(renderer.router, prefix="/api/v1")
app.include_router(story.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Micro-Drama API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
