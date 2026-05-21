import subprocess
import json
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    @staticmethod
    def extract_metadata(file_path: str) -> dict:
        """
        Extracts video metadata using ffprobe.
        Returns a dictionary containing duration, fps, resolution, etc.
        """
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration",
            "-of", "json",
            file_path
        ]
        
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            # Format the duration as requested (HH:MM:SS)
            raw_duration = float(data.get("format", {}).get("duration", 0))
            hours = int(raw_duration // 3600)
            minutes = int((raw_duration % 3600) // 60)
            seconds = int(raw_duration % 60)
            formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Calculate FPS from fraction (e.g., "30000/1001")
            fps_raw = data.get("streams", [{}])[0].get("r_frame_rate", "0/1")
            num, den = map(int, fps_raw.split('/'))
            fps = num / den if den != 0 else 0
            
            stream = data.get("streams", [{}])[0]

            return {
                "duration": formatted_duration,
                "fps": round(fps, 2),
                "resolution": f"{stream.get('width', 0)}x{stream.get('height', 0)}",
                "codec": stream.get("codec_name", "unknown")
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract metadata: {e.stderr}")
            raise Exception("Metadata extraction failed")

    @staticmethod
    def standardize_video(input_path: str, output_path: str):
        """
        Converts the video to H.264, AAC, 1080p, constant 30fps.
        """
        command = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",       # H.264 codec
            "-preset", "fast",       # Encoding speed
            "-crf", "23",            # Constant Rate Factor (Quality)
            "-c:a", "aac",           # AAC audio codec
            "-b:a", "128k",          # Audio bitrate
            "-vf", "scale=-1:1080",  # Scale to 1080p height, keeping aspect ratio
            "-r", "30",              # Constant 30 FPS
            "-y",                    # Overwrite output
            output_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Video standardization failed: {e.stderr.decode()}")
            raise Exception("Video standardization failed")
