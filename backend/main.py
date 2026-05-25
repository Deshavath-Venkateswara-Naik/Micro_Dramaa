from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingestion, segmentation, audio, speech, face, emotion, bgm, virality, nostalgia, drama, fusion, sequencer, renderer, story
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

app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(segmentation.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
app.include_router(speech.router, prefix="/api/v1")
app.include_router(face.router, prefix="/api/v1")
app.include_router(emotion.router, prefix="/api/v1")
app.include_router(bgm.router, prefix="/api/v1")
app.include_router(virality.router, prefix="/api/v1")
app.include_router(nostalgia.router, prefix="/api/v1")
app.include_router(drama.router, prefix="/api/v1")
app.include_router(fusion.router, prefix="/api/v1")
app.include_router(sequencer.router, prefix="/api/v1")
app.include_router(renderer.router, prefix="/api/v1")
app.include_router(story.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Micro-Drama API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
