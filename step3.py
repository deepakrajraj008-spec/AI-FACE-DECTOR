import cv2
import os

# Ask who is being enrolled (use 1 for yourself, 2 for a friend, ...)
person_id = input("Enter a numeric ID (1, 2, 3...): ").strip()
folder = os.path.join("dataset", person_id)
os.makedirs(folder, exist_ok=True)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

SHOTS = 40      # how many photos to save
count = 0       # photos saved so far
frame_no = 0    # counts frames so we don't save near-identical photos

while count < SHOTS:
    ok, frame = cam.read()
    if not ok:
        print("Could not read from the camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        frame_no += 1

        # Save every 3rd frame, so photos differ a little from each other
        if frame_no % 3 == 0:
            face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face)
            count += 1
        break   # only use the first face in the frame

    cv2.putText(frame, f"Photos: {count}/{SHOTS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.imshow("Step 3 - Enroll", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print(f"Saved {count} photos in {folder}")