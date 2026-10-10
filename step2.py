import cv2

# Open the webcam (0 = first camera)
cam = cv2.VideoCapture(0)

# Load OpenCV's ready-made face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ok, frame = cam.read()
    if not ok:
        print("Could not read from the camera")
        break

    # The detector works on a black-and-white copy of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces: returns a list of (x, y, width, height) boxes
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    # Draw a green box around each face found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Step 2 - Face detection", frame)

    # Press q to quit
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()