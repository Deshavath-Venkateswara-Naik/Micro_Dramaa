from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.emotion_processor import EmotionIntelligenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/emotion",
    tags=["Stage 6 - Emotion Intelligence"]
)

class EmotionRequest(BaseModel):
    video_id: str
    scene_metadata_path: str

@router.post("/process")
async def process_emotion(request: EmotionRequest):
    """
    Stage 6: Emotion Intelligence Engine
    Condenses raw face data into cinematic impact metrics using Gemini.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = EmotionIntelligenceProcessor(str(output_base_dir))
        results = processor.process_emotion(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 6 Emotion Intelligence completed",
            "video_id": request.video_id,
            "processed_scenes": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in emotion processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
