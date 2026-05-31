import json
import logging
import re
import subprocess
from pathlib import Path
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

class SmartClipRenderer:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.render_dir = self.output_base_dir / "render"
        self.render_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _time_to_seconds(t) -> float:
        """Converts 'HH:MM:SS.mmm' / 'MM:SS' / numeric to float seconds."""
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        try:
            parts = str(t).split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return float(t)
        except Exception:
            return 0.0

    @staticmethod
    def _get_duration(video_path: Path) -> float:
        """Reads video duration with ffprobe (no full decode)."""
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return float(out.stdout.strip())
        except Exception:
            return 0.0

    def _render_scenes(self, video_id: str, scenes: list, min_duration: float = 1.0) -> list:
        """
        Renders one clip per scene from scenes_and_plot.json using ffmpeg directly.
        Cuts from standardized_video.mp4 using each scene's start_time/end_time.

        Uses ffmpeg subprocess (not MoviePy) because MoviePy's subprocess wrapper
        intermittently raises "no attribute 'stdout'" and re-reads the whole file
        for every clip. ffmpeg with fast input seeking is reliable and fast.
        """
        video_path = self.output_base_dir / "standardized_video.mp4"
        if not video_path.exists():
            logger.error(f"Standardized video missing: {video_path}")
            return [{"error": f"Source video not found: {video_path}", "status": "failed"}]

        scenes_dir = self.render_dir
        scenes_dir.mkdir(parents=True, exist_ok=True)

        duration = self._get_duration(video_path)
        render_results = []

        for scene in scenes:
            num = scene.get("scene_number")
            start = self._time_to_seconds(scene.get("start_time"))
            end = self._time_to_seconds(scene.get("end_time"))

            # Clamp to the real video duration
            if duration > 0:
                start = max(0.0, min(start, duration))
                end = max(0.0, min(end, duration))
            clip_len = end - start

            if clip_len < min_duration:
                logger.info(f"Skipping scene {num}: too short ({clip_len:.2f}s).")
                render_results.append({
                    "scene_number": num,
                    "status": "skipped",
                    "reason": "duration below minimum"
                })
                continue

            setting = scene.get("setting", "Scene")
            setting_clean = re.sub(r'[^a-zA-Z0-9_]', '', str(setting).replace(" ", "_"))[:40]
            out_file = scenes_dir / f"scene_{int(num):03d}_{setting_clean}.mp4"

            # -ss before -i = fast seek; -t = exact duration; re-encode for frame-accurate cuts
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_path),
                "-t", f"{clip_len:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_non_negative",
                str(out_file)
            ]

            logger.info(f"Rendering scene {num}: {start:.2f}s -> {end:.2f}s ({clip_len:.2f}s)")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
                render_results.append({
                    "scene_number": num,
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "setting": scene.get("setting"),
                    "description": scene.get("description"),
                    "rendered_path": str(out_file),
                    "rendered_url": f"/storage/{self.output_base_dir.name}/render/{out_file.name}",
                    "status": "success"
                })
            else:
                err = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown ffmpeg error"]
                logger.error(f"Failed to render scene {num}: {err[0]}")
                render_results.append({
                    "scene_number": num,
                    "error": err[0],
                    "status": "failed"
                })

        return render_results

    def process_render(self, video_id: str, scene_metadata_path: str) -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # NEW: If the metadata is a scenes_and_plot.json (contains a "Scenes" list),
        # render one clip per scene directly from it.
        scenes = metadata.get("Scenes") if isinstance(metadata, dict) else None
        if scenes:
            logger.info(f"Rendering {len(scenes)} scenes from scenes_and_plot.json for {video_id}.")
            return self._render_scenes(video_id, scenes)

        render_results = []
        
        # Check for Master Series Roadmap
        roadmap_path = self.output_base_dir / "sequencer" / "master_series_sequence.json"
        roadmap = self._read_json(roadmap_path) if roadmap_path.exists() else {}
        episodes = roadmap.get("episodic_series", [])
        
        video_path = self.output_base_dir / "standardized_video.mp4"
        if not video_path.exists():
            logger.error(f"Standardized video missing: {video_path}")
            return []
            
        if episodes:
            logger.info("Master Series Roadmap found. Rendering all episodes.")
            
            # Render all episodes
            for idx, ep in enumerate(episodes):
                ep_num = ep.get("episode_number", idx + 1)
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
                
                title = ep.get("binge_worthy_title", ep.get("episode_title", f"episode_{ep_num}"))
                title_clean = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(" ", "_").replace("?", ""))
                out_file = self.render_dir / f"EP{int(ep_num):02d}_{title_clean}.mp4"
                
                try:
                    logger.info(f"Loading video and extracting Episode {ep_num} from {clip_start} to {clip_end}")
                    clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
                    
                    # 1. Sound Boost
                    clip = clip.volumex(1.5)
                    
                    # Keeping original "normal" format (16:9), no cropping
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

        return render_results

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}
