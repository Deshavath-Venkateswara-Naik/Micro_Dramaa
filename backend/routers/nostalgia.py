from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.nostalgia_processor import NostalgiaIntelligenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/nostalgia",
    tags=["Stage 10 - Nostalgia Intelligence"]
)

class NostalgiaRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process")
async def process_nostalgia(request: NostalgiaRequest):
    """
    Stage 10: Run Nostalgia Intelligence Engine on all scenes.
    """
    logger.info(f"Starting Nostalgia Intelligence for {request.video_id}")
    
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = NostalgiaIntelligenceProcessor(str(output_base_dir))
        results = processor.process_nostalgia(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 10 Nostalgia Intelligence completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in nostalgia processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
