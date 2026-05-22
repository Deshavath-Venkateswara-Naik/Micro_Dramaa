from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import logging

from services.emotion_processor import EmotionIntelligenceProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class EmotionRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

def process_emotion_background(video_id: str, scene_metadata_path: str, output_base_dir: str):
    try:
        processor = EmotionIntelligenceProcessor(output_base_dir)
        processor.process_emotion(video_id, scene_metadata_path)
    except Exception as e:
        logger.error(f"Error in background emotion processing: {e}")

@router.post("/emotion/process")
async def process_emotion(request: EmotionRequest):
    """
    Stage 6: Emotion Intelligence Engine
    Fuses multimodal signals from face, speech, and audio to track cinematic emotional curves.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = EmotionIntelligenceProcessor(str(output_base_dir))
        results = processor.process_emotion(request.video_path, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 6 Emotion Intelligence completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in emotion processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
