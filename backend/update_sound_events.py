import json
import logging
from services.script_formatter import CinematicScriptFormatter

logging.basicConfig(level=logging.INFO)

movie_dir = "/home/venkateswara/Micro_Drama/storage/MOV_43960E68"
json_path = f"{movie_dir}/script_transcript.json"
audio_path = f"{movie_dir}/audio/full_audio.wav"
txt_path = f"{movie_dir}/script_transcript.txt"

formatter = CinematicScriptFormatter(movie_dir)

# Load existing data
with open(json_path, 'r') as f:
    data = json.load(f)

dialogues = data.get("dialogues", [])

# Re-run ONLY sound event detection with the new threshold logic
print("Re-running Sound Event Detection with updated thresholds...")
new_sound_events = formatter.detect_sound_events(audio_path)

# Update JSON
data["sound_events"] = new_sound_events
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# Re-generate text script
print("Re-generating text script...")
script_text = formatter.generate_script(dialogues, new_sound_events)
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(script_text)

print("Successfully updated sound_events and overrode the existing transcript files!")
