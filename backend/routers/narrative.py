from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import logging

from services.narrative_processor import NarrativeIntelligenceProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class NarrativeRequest(BaseModel):
    video_id: str
    scene_metadata_path: str

def process_narrative_background(video_id: str, scene_metadata_path: str, output_base_dir: str):
    try:
        processor = NarrativeIntelligenceProcessor(output_base_dir)
        processor.process_narrative(video_id, scene_metadata_path)
    except Exception as e:
        logger.error(f"Error in background narrative processing: {e}")

@router.post("/narrative/process")
async def process_narrative(request: NarrativeRequest):
    """
    Stage 8: Narrative Intelligence Engine
    The Core Story Brain. Connects scenes, parses long-term story memory, and detects cinematic payoff arcs.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = NarrativeIntelligenceProcessor(str(output_base_dir))
        results = processor.process_narrative(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 8 Narrative Intelligence completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in narrative processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
