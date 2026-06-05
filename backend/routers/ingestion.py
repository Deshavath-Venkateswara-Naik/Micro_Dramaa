import uuid
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.storage import StorageService
from services.video_processor import VideoProcessor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class UrlIngestRequest(BaseModel):
    url: str
    language: str = "unknown"
    title: str = None

@router.post("/ingest/url")
async def ingest_video_url(request: UrlIngestRequest):
    """
    Ingests a video from a given URL, standardizes it, and extracts metadata.
    """
    video_id = f"MOV_{uuid.uuid4().hex[:8].upper()}"
    
    try:
        import yt_dlp
        
        # Determine paths
        processed_path = StorageService.get_processed_path(video_id)
        video_dir = os.path.dirname(processed_path)
        raw_file_path = os.path.join(video_dir, "raw_video.mp4")
        
        # Download the video
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': raw_file_path,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading video from {request.url}")
            ydl.download([request.url])
            
        # 3. Extract Metadata
        metadata = VideoProcessor.extract_metadata(raw_file_path)
        
        # Save source URL for later use (e.g. Subtitle API)
        url_file_path = os.path.join(video_dir, "source_url.txt")
        with open(url_file_path, "w") as f:
            f.write(request.url)
            
        # 4. Standardize Video (H.264, AAC, 1080p, 30fps)
        VideoProcessor.standardize_video(raw_file_path, processed_path)
        
        return {
            "video_id": video_id,
            "language": request.language,
            "duration": metadata.get("duration"),
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"URL ingestion failed for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
