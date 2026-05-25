import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.multimodal_fusion import MultimodalFusionProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Multimodal Fusion Intelligence"]
)

class FusionProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-fusion")
async def process_fusion(request: FusionProcessRequest):
    """
    Multimodal Fusion Intelligence Engine
    Combines outputs from Speech, Emotion, Face, and BGM processors into a single LLM call per scene.
    """
    try:
        video_path = Path(request.video_path)
        metadata_path = Path(request.scene_metadata_path)
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found at {request.scene_metadata_path}")
            
        output_dir = metadata_path.parent
        
        processor = MultimodalFusionProcessor(str(output_dir))
        
        results = processor.process_fusion(request.video_id, request.scene_metadata_path)
        
        return {
            "status": "success",
            "message": f"Processed {len(results)} scenes through Multimodal Fusion.",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Multimodal Fusion Engine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
