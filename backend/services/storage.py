import os
import shutil
from fastapi import UploadFile

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "storage")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

class StorageService:
    @staticmethod
    def save_upload_file(upload_file: UploadFile, video_id: str) -> str:
        """
        Saves the uploaded file to local storage.
        Returns the absolute path to the saved file.
        """
        # Create a directory for this specific video
        video_dir = os.path.join(STORAGE_DIR, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        # Save original file
        file_extension = os.path.splitext(upload_file.filename)[1]
        raw_file_path = os.path.join(video_dir, f"raw_video{file_extension}")
        
        with open(raw_file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return raw_file_path

    @staticmethod
    def get_processed_path(video_id: str) -> str:
        """
        Returns the path where the standardized video should be saved.
        """
        video_dir = os.path.join(STORAGE_DIR, video_id)
        os.makedirs(video_dir, exist_ok=True)
        return os.path.join(video_dir, "standardized_video.mp4")
