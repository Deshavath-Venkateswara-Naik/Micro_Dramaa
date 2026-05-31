import json
with open('storage/MOV_993D77C0/scenes_and_plot_raw.txt', 'r') as f:
    response_text = f.read()

last_brace_idx = response_text.rfind('    }')
if last_brace_idx != -1:
    cut_idx = last_brace_idx + 5
else:
    cut_idx = response_text.rfind('}') + 1

recovered_text = response_text[:cut_idx] + '\n  ],\n  "plot_of_the_movie": "Plot generation was truncated because the video is too long and the model hit the max output limit."\n}'

with open('recovered.json', 'w') as f:
    f.write(recovered_text)

try:
    json.loads(recovered_text)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
