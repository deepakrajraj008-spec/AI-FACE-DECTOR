import cv2
import os
import numpy as np

images = []   # the face photos
labels = []   # the person ID for each photo

# Go through every folder in dataset (the folder name is the person's ID)
for person_id in os.listdir("dataset"):
    folder = os.path.join("dataset", person_id)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(int(person_id))

if not images:
    print("No photos found. Run step3.py first.")
else:
    # Create the recognizer and teach it: these photos belong to these IDs
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    # Save what it learned
    recognizer.write("model.yml")
    print(f"Training complete: {len(images)} photos, {len(set(labels))} people")