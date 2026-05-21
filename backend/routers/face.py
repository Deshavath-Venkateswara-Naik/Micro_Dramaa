import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.face_processor import FaceIntelligenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Face Intelligence"]
)

class FaceProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-faces")
async def process_faces(request: FaceProcessRequest):
    """
    Stage 5: Cinematic Face Intelligence Engine
    Extracts frames, analyzes facial emotions using DeepFace, and calculates cinematic scores.
    """
    try:
        video_path = Path(request.video_path)
        metadata_path = Path(request.scene_metadata_path)
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found at {request.video_path}")
            
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found at {request.scene_metadata_path}")
            
        # The storage directory is typically /path/to/storage/MOV_ID/
        output_dir = metadata_path.parent
        
        processor = FaceIntelligenceProcessor(str(output_dir))
        
        results = processor.process_faces(
            str(video_path),
            str(metadata_path)
        )
        
        return {
            "status": "completed",
            "message": f"Stage 5 Face Intelligence completed for {request.video_id}",
            "output_dir": str(output_dir),
            "processed_scenes": len(results),
            "face_intelligence_results": results
        }
        
    except Exception as e:
        logger.error(f"Stage 5 processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
