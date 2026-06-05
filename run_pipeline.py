import argparse
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000/api/v1"

def print_step(step_num, name):
    print(f"\n{'='*50}")
    print(f"STEP {step_num}: {name}")
    print(f"{'='*50}")

def run_step(url, payload, step_name):
    print(f"Executing: POST {url}")
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"❌ ERROR in {step_name}: HTTP {response.status_code}")
            print(response.text)
            sys.exit(1)
        
        data = response.json()
        print(f"✅ SUCCESS: {step_name}")
        return data
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Could not connect to {url}. Is the backend running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Micro-Drama Pipeline Runner")
    parser.add_argument("--url", required=True, help="YouTube or Video URL to process")
    parser.add_argument("--language", default="Hindi", help="Language for micro-drama generation (e.g., Hindi, Telugu)")
    args = parser.parse_args()

    # --- Phase 1: Ingestion & Baseline ---
    print_step(1, "Ingestion (/ingest/url)")
    ingest_payload = {"url": args.url, "language": args.language, "title": "Auto-ingested"}
    ingest_result = run_step(f"{BASE_URL}/ingest/url", ingest_payload, "Ingestion")
    
    video_id = ingest_result["video_id"]
    print(f"Extracted Video ID: {video_id}")
    
    # Construct Paths needed for subsequent API calls
    storage_base = f"/home/venkateswara/Micro_Drama/storage/{video_id}"
    video_path = f"{storage_base}/standardized_video.mp4"
    scene_metadata_path = f"{storage_base}/scene_metadata.json"

    print_step(2, "Segmentation (/segment)")
    segment_payload = {"video_id": video_id}
    run_step(f"{BASE_URL}/segment", segment_payload, "Segmentation")

    # --- Phase 2: Feature Extraction ---
    print_step(3, "Audio Energy Processing (/process-audio)")
    audio_payload = {
        "video_id": video_id,
        "video_path": video_path,
        "scene_metadata_path": scene_metadata_path
    }
    run_step(f"{BASE_URL}/process-audio", audio_payload, "Audio Processing")

    print_step(4, "Sound Event Extraction (/extract-sound-events)")
    sound_event_payload = {
        "video_id": video_id,
        "video_path": video_path
    }
    run_step(f"{BASE_URL}/extract-sound-events", sound_event_payload, "Sound Events")

    print_step(5, "CLIP Embeddings (/process-clip-embeddings)")
    clip_payload = {
        "video_id": video_id,
        "video_path": video_path
    }
    run_step(f"{BASE_URL}/process-clip-embeddings", clip_payload, "CLIP Embeddings")

    print_step(6, "Speech & Emotion Extraction (/process-speech)")
    # Assuming the Sarvam language code is te-IN or hi-IN based on arg
    sarvam_lang = "hi-IN" if args.language.lower() == "hindi" else "te-IN"
    speech_payload = {
        "video_id": video_id,
        "video_path": video_path,
        "scene_metadata_path": scene_metadata_path,
        "language": sarvam_lang
    }
    run_step(f"{BASE_URL}/process-speech", speech_payload, "Speech Extraction")

    # --- Phase 3: Data Fusion ---
    print_step(7, "Gemini LLM Intelligence Fusion (/gemini-llm)")
    gemini_payload = {"video_id": video_id}
    run_step(f"{BASE_URL}/gemini-llm", gemini_payload, "Gemini Fusion")

    # --- Phase 4: Generation & Rendering ---
    print_step(8, "Microdrama Generation - Pass 1 (/generate-microdrama)")
    gen1_payload = {"video_id": video_id, "language": args.language}
    run_step(f"{BASE_URL}/generate-microdrama", gen1_payload, "Generation Pass 1")

    print_step(9, "Microdrama Generation - Pass 2 (Duration Enforcement) (/generate-microdrama2)")
    gen2_payload = {"video_id": video_id, "language": args.language}
    run_step(f"{BASE_URL}/generate-microdrama2", gen2_payload, "Generation Pass 2")

    print_step(10, "Final Rendering (/render-microdrama)")
    render_payload = {"video_id": video_id}
    run_step(f"{BASE_URL}/render-microdrama", render_payload, "Rendering")

    print(f"\n🎉 PIPELINE COMPLETE FOR {video_id} 🎉")
    print(f"Check your /storage/{video_id}/render directory for the final micro-dramas!")

if __name__ == "__main__":
    main()
