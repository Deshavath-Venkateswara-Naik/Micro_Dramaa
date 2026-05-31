import logging
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class AudioChunker:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        
    def chunk_audio(self, audio_path: str, shot_id: str, max_duration_ms: int = 28000) -> list:
        """
        Splits audio intelligently to ensure no chunk exceeds max_duration_ms (28s).
        It searches backward for the quietest moment (silence) to avoid cutting words in half.
        Returns a list of metadata for each chunk containing the global offset.
        """
        logger.info(f"Chunking audio for {shot_id} at {audio_path}")
        
        try:
            audio = AudioSegment.from_wav(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio for chunking: {e}")
            return []
            
        chunks_meta = []
        total_duration_ms = len(audio)
        current_start = 0
        chunk_idx = 1
        
        while current_start < total_duration_ms:
            # Determine the maximum end time for this chunk
            end_bound = min(current_start + max_duration_ms, total_duration_ms)
            
            # If we reached the end, just take the rest
            if end_bound == total_duration_ms:
                actual_end = end_bound
            else:
                # Look backwards from end_bound for silence to avoid cutting words
                # We analyze the last 5 seconds of the allowed window
                search_start = max(current_start, end_bound - 5000)
                window = audio[search_start:end_bound]
                
                best_split_offset = end_bound
                if len(window) >= 200:
                    min_rms = float('inf')
                    best_split = len(window)
                    
                    step = 50 # 50ms steps
                    for i in range(0, len(window) - 200, step):
                        segment_rms = window[i:i+200].rms
                        if segment_rms < min_rms:
                            min_rms = segment_rms
                            best_split = i + 100 # Split in the middle of the quietest part
                    
                    # If the quietest part is reasonably quiet, use it
                    if min_rms < (audio.rms * 0.7):
                        best_split_offset = search_start + best_split
                
                actual_end = best_split_offset
                
            # Extract and save chunk
            chunk_audio = audio[current_start:actual_end]
            chunk_path = self.chunk_dir / f"{shot_id}_chunk_{chunk_idx:03d}.wav"
            chunk_audio.export(str(chunk_path), format="wav")
            
            chunks_meta.append({
                "chunk_id": f"{shot_id}_CHK{chunk_idx:02d}",
                "file_path": str(chunk_path),
                "global_offset_ms": current_start,
                "duration_ms": actual_end - current_start
            })
            
            current_start = actual_end
            chunk_idx += 1
            
        logger.info(f"Generated {len(chunks_meta)} chunks for {shot_id}")
        return chunks_meta
