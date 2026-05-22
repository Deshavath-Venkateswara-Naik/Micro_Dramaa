from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.drama_scorer_processor import DramaScoringProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/drama",
    tags=["Stage 11 - Drama Scoring Engine"]
)

class DramaRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str
    genre: str = "Action"

@router.post("/process")
async def process_drama(request: DramaRequest):
    """
    Stage 11: Run Multi-Layer Drama Scoring Engine on all scenes.
    """
    logger.info(f"Starting Drama Scoring Intelligence for {request.video_id} with Genre {request.genre}")
    
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = DramaScoringProcessor(str(output_base_dir))
        results = processor.process_drama_scoring(request.video_id, request.scene_metadata_path, request.genre)
        
        return {
            "status": "completed",
            "message": "Stage 11 Drama Scoring Engine completed",
            "video_id": request.video_id,
            "genre": request.genre,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in drama scoring processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
