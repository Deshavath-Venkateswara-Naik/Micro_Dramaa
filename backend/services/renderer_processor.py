import os
import json
import logging
import re
from pathlib import Path
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import moviepy.video.fx.all as vfx

logger = logging.getLogger(__name__)

class SmartClipRenderer:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.render_dir = self.output_base_dir / "rendered"
        self.render_dir.mkdir(parents=True, exist_ok=True)

    def process_render(self, video_id: str, scene_metadata_path: str) -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        render_results = []
        
        # Check for Master Series Roadmap
        roadmap_path = self.output_base_dir / "sequencer" / "master_series_sequence.json"
        roadmap = self._read_json(roadmap_path) if roadmap_path.exists() else {}
        episodes = roadmap.get("episodic_series", {}).get("episodes", [])
        
        video_path = self.output_base_dir / "standardized_video.mp4"
        if not video_path.exists():
            logger.error(f"Standardized video missing: {video_path}")
            return []
            
        if episodes:
            logger.info("Master Series Roadmap found. Rendering all episodes.")
            
            # Render all episodes
            for ep in episodes:
                ep_num = ep.get("episode_number", 1)
                # Use explicit start_time and end_time if available, else fallback to extracting from candidate_reference
                clip_start = ep.get("start_time")
                clip_end = ep.get("end_time")
                
                if not clip_start:
                    candidate_ref = ep.get("candidate_reference", "")
                    match = re.search(r'start_time:\s*(\d{2}:\d{2}:\d{2})', candidate_ref)
                    if match:
                        clip_start = match.group(1)
                    else:
                        logger.warning(f"Could not find start_time for episode {ep_num}, skipping.")
                        continue
                        
                if not clip_end:
                    # Convert clip_start to seconds and add 60 for clip_end
                    parts = clip_start.split(':')
                    s_seconds = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                    e_seconds = s_seconds + 60
                    clip_end = f"{e_seconds//3600:02d}:{(e_seconds%3600)//60:02d}:{e_seconds%60:02d}"
                title = ep.get("episode_title", f"episode_{ep_num}")
                title_clean = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(" ", "_").replace("?", ""))
                out_file = self.render_dir / f"EP{int(ep_num):02d}_{title_clean}.mp4"
                
                try:
                    logger.info(f"Loading video and extracting Episode {ep_num} from {clip_start} to {clip_end}")
                    clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
                    
                    # 1. Sound Boost
                    clip = clip.volumex(1.5)
                    
                    # Normal format requested (16:9), no cropping to 9:16
                    logger.info("Keeping original video aspect ratio for master_series.")
                    
                    # Render the final file
                    logger.info(f"Rendering final episodic reel to {out_file}...")
                    clip.write_videofile(
                        str(out_file), 
                        codec="libx264", 
                        audio_codec="aac", 
                        threads=4, 
                        preset="ultrafast",
                        logger=None
                    )
                    
                    clip.close()
                    
                    render_results.append({
                        "episode_number": ep_num,
                        "rendered_path": str(out_file),
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to render episode {ep_num}: {e}")
                    render_results.append({
                        "episode_number": ep_num,
                        "error": str(e),
                        "status": "failed"
                    })
        # Also render Story Candidates with 9:16 crop
        story_candidates_path = self.output_base_dir / "story" / "story_candidates.json"
        story_data = self._read_json(story_candidates_path) if story_candidates_path.exists() else {}
        candidates = story_data.get("microdrama_candidates", [])
        
        if candidates:
            logger.info("Story candidates found. Rendering with 9:16 Face Focus crop.")
            
            for idx, candidate in enumerate(candidates):
                clip_start = candidate.get("start_time")
                clip_end = candidate.get("end_time")
                title = candidate.get("binge_worthy_title", f"candidate_{idx}")
                title_clean = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(" ", "_"))
                
                if not clip_start or not clip_end:
                    continue
                    
                out_file = self.render_dir / f"story_candidate_{idx}_{title_clean}.mp4"
                
                try:
                    logger.info(f"Loading video and extracting candidate {idx} from {clip_start} to {clip_end}")
                    clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
                    
                    # 1. Sound Boost
                    clip = clip.volumex(1.5)
                    
                    # 2. Face Focus (Crop to 9:16)
                    w, h = clip.size
                    target_w = int(h * 9 / 16)
                    x_center = w / 2
                    x1 = max(0, x_center - (target_w / 2))
                    x2 = min(w, x_center + (target_w / 2))
                    
                    logger.info(f"Cropping candidate to 9:16 ratio (width: {target_w}, height: {h})")
                    clip = vfx.crop(clip, x1=x1, y1=0, x2=x2, y2=h)
                    
                    # 3. Render the final file
                    logger.info(f"Rendering story candidate to {out_file}...")
                    clip.write_videofile(
                        str(out_file), 
                        codec="libx264", 
                        audio_codec="aac", 
                        threads=4, 
                        preset="ultrafast",
                        logger=None
                    )
                    
                    clip.close()
                    
                    render_results.append({
                        "type": "story_candidate",
                        "candidate_index": idx,
                        "title": title,
                        "rendered_path": str(out_file),
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to render candidate {idx}: {e}")
                    render_results.append({
                        "type": "story_candidate",
                        "candidate_index": idx,
                        "error": str(e),
                        "status": "failed"
                    })

        return render_results

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}
