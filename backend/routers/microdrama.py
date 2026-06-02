import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.microdrama_generator import MicrodramaGenerator
from services.microdrama_renderer import MicrodramaRenderer

router = APIRouter()

class MicrodramaRequest(BaseModel):
    video_id: str
    language: str = "Hindi"

@router.post("/generate-microdrama", tags=["Microdrama Generator"])
async def generate_microdrama(request: MicrodramaRequest):
    """
    Generates strictly 30-100 second microdrama candidates for a video by
    consuming its `scenes_and_plot.json` and `dialogue_diarization.json`
    artifacts. The LLM proposes creative selections.
    """
    try:
        storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "storage", request.video_id)
        if not os.path.exists(storage_path):
            raise HTTPException(status_code=404, detail=f"Storage directory not found for video {request.video_id}")

        generator = MicrodramaGenerator(output_base_dir=storage_path)
        result = generator.generate(video_id=request.video_id, language=request.language)

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RenderRequest(BaseModel):
    video_id: str


@router.post("/render-microdrama", tags=["Microdrama Generator"])
async def render_microdrama(request: RenderRequest):
    """
    Renders microdrama candidate videos using ffmpeg from the candidates JSON list.
    """
    try:
        storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "storage", request.video_id)
        if not os.path.exists(storage_path):
            raise HTTPException(status_code=404, detail=f"Storage directory not found for video {request.video_id}")

        renderer = MicrodramaRenderer(output_base_dir=storage_path)
        result = renderer.render(video_id=request.video_id)

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
