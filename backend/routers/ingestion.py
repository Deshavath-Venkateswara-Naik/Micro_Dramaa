import uuid
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from services.storage import StorageService
from services.video_processor import VideoProcessor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest")
async def ingest_video(
    video: UploadFile = File(...),
    language: str = Form("unknown"),
    title: str = Form(None),
):
    """
    Ingests a raw video file, standardizes it, and extracts metadata.
    """
    # 1. Generate unique video ID
    video_id = f"MOV_{uuid.uuid4().hex[:8].upper()}"
    
    try:
        # 2. Save the uploaded file locally
        raw_file_path = StorageService.save_upload_file(video, video_id)
        
        # 3. Extract Metadata
        metadata = VideoProcessor.extract_metadata(raw_file_path)
        
        # 4. Standardize Video (H.264, AAC, 1080p, 30fps)
        processed_path = StorageService.get_processed_path(video_id)
        
        # Note: In a production environment, this step should be sent to a background worker (e.g., Celery)
        # For this stage, we process it synchronously.
        VideoProcessor.standardize_video(raw_file_path, processed_path)
        
        # 5. Return expected output
        return {
            "video_id": video_id,
            "language": language,
            "duration": metadata.get("duration"),
            "metadata": metadata # Including extra metadata for debugging
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
