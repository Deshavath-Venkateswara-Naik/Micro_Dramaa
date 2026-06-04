import requests

url = "http://localhost:8000/api/v1/extract-sound-events"
payload = {
    "video_id": "MOV_43960E68",
    "video_path": "/home/venkateswara/Micro_Drama/storage/MOV_43960E68/MOV_43960E68_global_final_merged.mp4"
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
