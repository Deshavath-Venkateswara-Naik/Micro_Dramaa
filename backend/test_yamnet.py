import tensorflow_hub as hub
import csv

model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode('utf-8')

classes = []
with open(class_map_path) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        classes.append(row[2])
        
print([c for c in classes if "Music" in c or "Door" in c or "Laugh" in c or "Cry" in c or "Applause" in c or "Siren" in c])
