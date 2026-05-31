import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.story_engine import StoryIntelligenceEngine

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Story Intelligence"]
)

class StoryProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-story")
async def process_story(request: StoryProcessRequest):
    """
    Layer 3: Story Intelligence Engine
    Aggregates all Layer 2 signals and generates microdrama candidates using Gemini 2.5 Pro.
    """
    try:
        video_path = Path(request.video_path)
        metadata_path = Path(request.scene_metadata_path)
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found at {request.video_path}")
            
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found at {request.scene_metadata_path}")
            
        output_dir = metadata_path.parent
        
        engine = StoryIntelligenceEngine(str(output_dir))
        
        results = engine.process_story(request.video_id, request.video_path, request.scene_metadata_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Layer 3 Story Engine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
