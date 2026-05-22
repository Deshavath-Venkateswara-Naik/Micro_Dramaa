from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.continuous_processor import ContinuousIntelligenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/continuous",
    tags=["Stage 14 - Continuous Micro-Drama Generation"]
)

class ContinuousRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process")
async def process_continuous(request: ContinuousRequest):
    """
    Stage 14: Generate the global Continuous Episodic Story Roadmap.
    """
    logger.info(f"Starting Continuous Micro-Drama Generation for {request.video_id}")
    
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = ContinuousIntelligenceProcessor(str(output_base_dir))
        results = processor.process_continuous_story(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 14 Continuous Micro-Drama Generation completed",
            "video_id": request.video_id,
            "episodes_generated": len(results),
            "episodes": results
        }
    except Exception as e:
        logger.error(f"Error in continuous processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
