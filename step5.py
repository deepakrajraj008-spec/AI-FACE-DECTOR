import cv2

# Change these to match the IDs you used in step 3
names = {
    1: "sudhanshu",
    2: "Friend",
}

THRESHOLD = 70   # lower distance = better match; you will tune this

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    if not ok:
        print("Could not read from the camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Same preparation as in enrollment: crop and resize to 200x200
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

        # Ask the model: who is this, and how far from the closest match?
        person_id, distance = recognizer.predict(face)

        if distance < THRESHOLD and person_id in names:
            label = names[person_id]
            color = (0, 200, 0)      # green = recognised
        else:
            label = "Unknown"
            color = (0, 0, 255)      # red = not recognised

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {distance:.0f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(label, round(distance, 1))

    cv2.imshow("Step 5 - Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()