import os
import json
import logging
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv("backend/.env")

def run_diarization(audio_path, transcript_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Load the existing chunked transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data.get("segments", [])
    
    # 2. Load audio
    logger.info("Loading audio...")
    audio_array = whisperx.load_audio(audio_path)
    
    # 3. Align chunks to get word-level timestamps
    logger.info("Aligning chunked transcript to get word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code="hi", device=device)
    align_result = whisperx.align(segments, model_a, metadata, audio_array, device, return_char_alignments=False)
    
    words = []
    for segment in align_result["segments"]:
        for word in segment.get("words", []):
            if "start" in word and "end" in word:
                words.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                })
    
    logger.info(f"Aligned {len(words)} words.")
    
    # 4. Run Pyannote Speaker Diarization
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in backend/.env")
        return
        
    logger.info("Running Speaker Diarization...")
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio_array)
    
    # 5. Merge words with speakers
    logger.info("Merging words with speakers...")
    dialogues = []
    current_speaker = None
    current_dialogue = None
    
    for word_obj in words:
        w_start = float(word_obj.get("start", 0))
        w_end = float(word_obj.get("end", 0))
        w_text = word_obj.get("word", "").strip()
        
        best_speaker = "UNKNOWN"
        max_overlap = 0
        
        for _, row in diarize_segments.iterrows():
            s_start = row['start']
            s_end = row['end']
            overlap = max(0, min(w_end, s_end) - max(w_start, s_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = row['speaker']
                
        if current_speaker != best_speaker or current_dialogue is None:
            if current_dialogue is not None:
                dialogues.append(current_dialogue)
                
            current_speaker = best_speaker
            current_dialogue = {
                "speaker": current_speaker,
                "start": round(w_start, 2),
                "end": round(w_end, 2),
                "text": w_text
            }
        else:
            current_dialogue["end"] = round(w_end, 2)
            current_dialogue["text"] += f" {w_text}"
            
            if w_text and w_text[-1] in ['.', '!', '?']:
                dialogues.append(current_dialogue)
                current_dialogue = None

    if current_dialogue is not None:
        dialogues.append(current_dialogue)
        
    # Save the result
    result_json = {"dialogues": dialogues}
    
    # ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully saved diarization to {output_path}")

if __name__ == "__main__":
    audio = "/home/venkateswara/Micro_Drama/storage/MOV_993D77C0/audio/dialogue/global_dialogue.wav"
    transcript = "/home/venkateswara/Micro_Drama/storage/MOV_993D77C0/audio/chunks/MOV_993D77C0_global_final_merged.json"
    output = "/home/venkateswara/Micro_Drama/storage/MOV_993D77C0/audio/dialogue_diarization.json"
    run_diarization(audio, transcript, output)
