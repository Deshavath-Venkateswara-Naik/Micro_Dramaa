from fastapi import FastAPI
from routers import ingestion, segmentation, audio, speech, face, emotion, bgm, narrative, virality
import uvicorn
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Micro-Drama AI Pipeline", description="Backend for Micro-Drama generation")

app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(segmentation.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
app.include_router(speech.router, prefix="/api/v1")
app.include_router(face.router, prefix="/api/v1")
app.include_router(emotion.router, prefix="/api/v1")
app.include_router(bgm.router, prefix="/api/v1")
app.include_router(narrative.router, prefix="/api/v1")
app.include_router(virality.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Micro-Drama API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
