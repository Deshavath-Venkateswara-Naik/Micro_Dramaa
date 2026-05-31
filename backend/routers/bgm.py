from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

from services.bgm_processor import BgmIntelligenceProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class BgmRequest(BaseModel):
    video_id: str
    scene_metadata_path: str

def process_bgm_background(video_id: str, scene_metadata_path: str, output_base_dir: str):
    try:
        processor = BgmIntelligenceProcessor(output_base_dir)
        processor.process_bgm(video_id, scene_metadata_path)
    except Exception as e:
        logger.error(f"Error in background BGM processing: {e}")

@router.post("/bgm/process")
async def process_bgm(request: BgmRequest):
    """
    Stage 7: BGM & Music Intelligence Engine
    Analyzes isolated BGM tracks and fuses them with Stage 6 emotions to map cinematic music curves.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = BgmIntelligenceProcessor(str(output_base_dir))
        results = processor.process_bgm(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 7 BGM Intelligence completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in BGM processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
