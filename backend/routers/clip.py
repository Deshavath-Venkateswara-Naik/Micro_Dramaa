from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
from services.clip_processor import ClipProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class ClipEmbeddingsRequest(BaseModel):
    video_id: str
    video_path: str
    
@router.post("/process-clip-embeddings")
async def process_clip_embeddings(request: ClipEmbeddingsRequest):
    """
    Extracts mid-frames from shots and calculates CLIP embeddings.
    Requires shots.json to already exist from Stage 2 Segmentation.
    """
    try:
        logger.info(f"Received request to process CLIP embeddings for {request.video_id}")
        output_dir = f"/home/venkateswara/Micro_Drama/storage/{request.video_id}"
        
        processor = ClipProcessor(output_base_dir=output_dir)
        results = processor.extract_embeddings(
            video_id=request.video_id,
            video_path=request.video_path
        )
        
        if results.get("status") == "error":
            raise HTTPException(status_code=500, detail=results["message"])
            
        return results

    except Exception as e:
        logger.error(f"Error in process-clip-embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
