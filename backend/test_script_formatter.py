import logging
from services.script_formatter import CinematicScriptFormatter
logging.basicConfig(level=logging.INFO)

formatter = CinematicScriptFormatter('/home/venkateswara/Micro_Drama/storage/MOV_43960E68')
fmt_res = formatter.process(
    '/home/venkateswara/Micro_Drama/storage/MOV_43960E68/dialogue_diarization.json',
    '/home/venkateswara/Micro_Drama/storage/MOV_43960E68/audio/full_audio.wav'
)
print("Formatter Result:", fmt_res)
