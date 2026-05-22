from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
from services.renderer_processor import SmartClipRenderer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/renderer",
    tags=["Stage 13 - Smart Clip Rendering Engine"]
)

class RendererRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process")
async def process_renderer(request: RendererRequest):
    """
    Stage 13: Render the final Micro-Drama reel using MoviePy.
    """
    logger.info(f"Starting Smart Clip Rendering Engine for {request.video_id}")
    
    metadata_path = Path(request.scene_metadata_path)
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="Scene metadata file not found")
        
    output_base_dir = metadata_path.parent
    
    try:
        processor = SmartClipRenderer(str(output_base_dir))
        results = processor.process_render(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "completed",
            "message": "Stage 13 Smart Clip Rendering Engine completed",
            "video_id": request.video_id,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in rendering processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
