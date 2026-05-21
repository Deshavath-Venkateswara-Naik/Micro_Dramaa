from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import logging

from services.virality_processor import ViralityIntelligenceProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class ViralityRequest(BaseModel):
    video_id: str
    scene_metadata_path: str

def process_virality_background(video_id: str, scene_metadata_path: str, output_base_dir: str):
    try:
        processor = ViralityIntelligenceProcessor(output_base_dir)
        processor.process_virality(video_id, scene_metadata_path)
    except Exception as e:
        logger.error(f"Error in background virality processing: {e}")

@router.post("/virality/process")
async def process_virality(request: ViralityRequest):
    """
    Stage 9: Virality & Audience Psychology Engine
    The Final Intelligence Layer. Predicts social media performance and dopamine triggers.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = ViralityIntelligenceProcessor(str(output_base_dir))
        results = processor.process_virality(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 9 Virality Intelligence completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in virality processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
