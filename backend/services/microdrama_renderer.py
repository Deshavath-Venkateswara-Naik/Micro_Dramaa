import os
import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class MicrodramaRenderer:
    """
    Reads the `microdrama_candidates.json` file and renders the individual
    microdrama clips from the standardized video using ffmpeg.
    """

    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)

    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitizes a candidate title to be used as a safe filename.
        Replaces spaces with underscores and removes non-alphanumeric characters.
        """
        if not title:
            return "untitled_clip"
        
        # Remove non-alphanumeric characters (except spaces and underscores)
        cleaned = re.sub(r'[^\w\s]', '', title)
        # Replace spaces with underscores
        cleaned = re.sub(r'\s+', '_', cleaned)
        
        return cleaned.strip('_').lower()

    def render(self, video_id: str) -> dict:
        """
        Renders all microdrama candidates for the given video_id.
        """
        candidates_file = self.output_base_dir / "microdrama_candidates.json"
        video_source = self.output_base_dir / "standardized_video.mp4"
        render_dir = self.output_base_dir / "microdrama-render"

        if not candidates_file.exists():
            return {"video_id": video_id, "status": "failed", "error": f"Missing file: {candidates_file}"}
        
        if not video_source.exists():
            return {"video_id": video_id, "status": "failed", "error": f"Missing video source: {video_source}"}

        # Create output directory
        render_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(candidates_file, 'r', encoding='utf-8') as f:
                envelope = json.load(f)
        except Exception as e:
            return {"video_id": video_id, "status": "failed", "error": f"Failed to read JSON: {e}"}

        candidates = envelope.get("microdrama_candidates", [])
        if not candidates:
            return {"video_id": video_id, "status": "failed", "error": "No candidates found in JSON"}

        rendered_clips = []
        errors = []

        for i, candidate in enumerate(candidates):
            title = candidate.get("title", f"clip_{i+1}")
            start_time = candidate.get("start_time")
            end_time = candidate.get("end_time")

            if not start_time or not end_time:
                errors.append(f"Candidate {title} is missing start_time or end_time.")
                continue

            safe_title = f"{i+1:02d}_{self._sanitize_filename(title)}"
            output_file = render_dir / f"{safe_title}.mp4"

            # Re-encode to ensure frame-accurate cuts and visible first frames
            command = [
                "ffmpeg",
                "-y", # Overwrite output files without asking
                "-ss", start_time,
                "-to", end_time,
                "-i", str(video_source),
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                str(output_file)
            ]

            try:
                # Run ffmpeg synchronously
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                rendered_clips.append({
                    "title": title,
                    "file_path": str(output_file)
                })
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8', errors='ignore')
                logger.error(f"Failed to render {title}: {error_msg}")
                errors.append(f"FFmpeg error on {title}")

        return {
            "video_id": video_id,
            "status": "completed",
            "rendered_clips": rendered_clips,
            "errors": errors
        }
