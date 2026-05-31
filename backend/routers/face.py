from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.face_processor import FaceIntelligenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Stage 5 - Face Intelligence"]
)

class FaceRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-faces")
async def process_faces(request: FaceRequest):
    """
    Stage 5: Face Intelligence Engine
    Analyzes visual performance in scenes using InsightFace and HSEmotion.
    """
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = FaceIntelligenceProcessor(str(output_base_dir))
        results = processor.process_faces(request.video_id, request.video_path, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 5 Face Intelligence completed",
            "video_id": request.video_id,
            "processed_scenes": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in face processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
