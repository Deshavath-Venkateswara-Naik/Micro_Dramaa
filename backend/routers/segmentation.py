import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from services.segmentation import SegmentationService
from services.storage import STORAGE_DIR
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class SegmentRequest(BaseModel):
    video_id: str

@router.post("/segment")
async def segment_video(req: SegmentRequest, background_tasks: BackgroundTasks):
    """
    Triggers the Stage 2 Cinematic Segmentation pipeline for a previously ingested video.
    """
    video_id = req.video_id
    video_dir = os.path.join(STORAGE_DIR, video_id)
    processed_path = os.path.join(video_dir, "standardized_video.mp4")
    
    if not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail=f"Processed video not found for {video_id}. Did Stage 1 complete?")
        
    try:
        # In a real production environment, this would definitely be a Celery task
        # because analyzing a full movie takes a long time. 
        # For this stage, we process synchronously or in a FastAPI background task to return a response quickly.
        
        # We will process synchronously here for immediate testing results.
        result = SegmentationService.process_video(video_id=video_id, video_path=processed_path)
        
        # Save the metadata to the storage directory
        metadata_path = os.path.join(video_dir, "scene_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    except Exception as e:
        logger.error(f"Segmentation failed for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
