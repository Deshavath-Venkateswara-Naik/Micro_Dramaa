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
        
        # Check for Stage 14 Continuous Roadmap
        roadmap_path = self.output_base_dir / "continuous" / "continuous_micro_drama_roadmap.json"
        roadmap = self._read_json(roadmap_path) if roadmap_path.exists() else {}
        episodes = roadmap.get("episodes", [])
        
        video_path = self.output_base_dir / "standardized_video.mp4"
        if not video_path.exists():
            logger.error(f"Standardized video missing: {video_path}")
            return []
            
        if episodes:
            logger.info("Continuous Episodic Roadmap found. Rendering Continuous Episodes.")
            scene_dict = {s["scene_id"]: s for s in scenes}
            
            # Render all episodes
            for ep in episodes:
                ep_num = ep.get("episode_number", 1)
                scenes_inc = ep.get("scenes_included", [])
                
                if not scenes_inc:
                    continue
                    
                first_scene = scene_dict.get(scenes_inc[0])
                last_scene = scene_dict.get(scenes_inc[-1])
                
                if not first_scene or not last_scene:
                    continue
                    
                # Time string format is "00:00:00" or similar, need to convert to seconds or use MoviePy parsing
                # MoviePy can accept "00:00:00" format!
                clip_start = first_scene.get("start", "00:00:00")
                clip_end = last_scene.get("end", "00:01:00")
                
                # Attempt to parse specific timestamps from trim_instructions
                trim_inst = ep.get("trim_instructions", "")
                times = re.findall(r'\d{2}:\d{2}:\d{2}', trim_inst)
                if len(times) >= 1:
                    clip_start = times[0]
                    # If a second timestamp exists, use it as end, else assume a 60s clip
                    if len(times) >= 2:
                        clip_end = times[1]
                    else:
                        # Convert clip_start to seconds and add 60
                        parts = clip_start.split(':')
                        s_seconds = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                        e_seconds = s_seconds + 60
                        clip_end = f"{e_seconds//3600:02d}:{(e_seconds%3600)//60:02d}:{e_seconds%60:02d}"
                
                out_file = self.render_dir / f"continuous_episode_{ep_num}.mp4"
                
                try:
                    logger.info(f"Loading video and extracting Episode {ep_num} from {clip_start} to {clip_end}")
                    clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
                    
                    # 1. Sound Boost
                    clip = clip.volumex(1.5)
                    
                    # Normal format requested (16:9), no cropping to 9:16
                    logger.info("Keeping original video aspect ratio")
                    
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
            return render_results

        # Fallback to single scene rendering if no episodes
        if not scenes:
            return []
            
        for scene in scenes[:1]:  # Process the first scene as an example
            scene_id = scene.get("scene_id")
            start_time = scene.get("start_time", 0)
            
            # Load Intelligence
            sequencer_data = self._read_json(self.output_base_dir / "sequencer" / f"sequencer_intelligence_{scene_id}.json")
            face_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            
            # Determine Timestamps from Sequencer
            hook = sequencer_data.get("hook", "0s-5s")
            cliffhanger = sequencer_data.get("cliffhanger", "5s-10s")
            
            # Extract numbers from string using regex e.g. "0s-3s (shocking)" -> 0, 3
            try:
                start_s = int(re.findall(r'(\d+)s', hook)[0])
                end_s = int(re.findall(r'(\d+)s', cliffhanger)[1])
            except (IndexError, ValueError):
                start_s = 0
                end_s = 15
                
            if end_s <= start_s:
                end_s = start_s + 15
            
            # Adjust global timestamps
            clip_start = start_time + start_s
            clip_end = start_time + end_s
            
            video_path = self.output_base_dir / "standardized_video.mp4"
            if not video_path.exists():
                logger.error(f"Standardized video missing: {video_path}")
                continue
                
            out_file = self.render_dir / f"final_micro_drama_{scene_id}.mp4"
            
            try:
                logger.info(f"Loading video and extracting from {clip_start}s to {clip_end}s")
                clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
                
                # 1. Sound Boost (increase volume by 1.5x)
                clip = clip.volumex(1.5)
                
                # 2. Face Focus (Crop to 9:16)
                # We do a simple center crop for now to guarantee 9:16 ratio safely
                w, h = clip.size
                target_w = int(h * 9 / 16)
                x_center = w / 2
                
                # If face intelligence exists, we could shift x_center.
                # Keeping it simple for stability:
                x1 = max(0, x_center - (target_w / 2))
                x2 = min(w, x_center + (target_w / 2))
                
                logger.info(f"Cropping to 9:16 ratio (width: {target_w}, height: {h})")
                clip = vfx.crop(clip, x1=x1, y1=0, x2=x2, y2=h)
                
                # 3. Cinematic Subtitles (Try to add a watermark/subtitle text)
                try:
                    # Attempt to add a text clip. This might fail if ImageMagick is missing.
                    txt_clip = TextClip("Micro-Drama Engine", fontsize=50, color='white', bg_color='black')
                    txt_clip = txt_clip.set_position('bottom').set_duration(clip.duration).margin(bottom=50, opacity=0)
                    
                    final_clip = CompositeVideoClip([clip, txt_clip])
                except Exception as text_e:
                    logger.warning(f"ImageMagick missing or TextClip failed. Skipping subtitles: {text_e}")
                    final_clip = clip
                
                # 4. Render the final file
                logger.info(f"Rendering final reel to {out_file}...")
                final_clip.write_videofile(
                    str(out_file), 
                    codec="libx264", 
                    audio_codec="aac", 
                    threads=4, 
                    preset="ultrafast",
                    logger=None # Suppress massive MoviePy output
                )
                
                clip.close()
                final_clip.close()
                
                render_results.append({
                    "scene_id": scene_id,
                    "rendered_path": str(out_file),
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Failed to render video for {scene_id}: {e}")
                render_results.append({
                    "scene_id": scene_id,
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
