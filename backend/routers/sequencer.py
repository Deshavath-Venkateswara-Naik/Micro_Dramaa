from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.sequencer_processor import DramaSequencerProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sequencer",
    tags=["Stage 12 - AI Drama Sequencer"]
)

class SequencerRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process")
async def process_sequencer(request: SequencerRequest):
    """
    Stage 12: Run AI Drama Sequencer on all scenes.
    """
    logger.info(f"Starting AI Drama Sequencer for {request.video_id}")
    
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = DramaSequencerProcessor(str(output_base_dir))
        results = processor.process_sequencer(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 12 AI Drama Sequencer completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in sequencer processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
