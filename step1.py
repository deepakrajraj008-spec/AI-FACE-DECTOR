import cv2

# Open the webcam (0 = first camera)
cam = cv2.VideoCapture(0)

while True:
    # Read one picture (frame) from the camera
    ok, frame = cam.read()
    if not ok:
        print("Could not read from the camera")
        break

    # Show the frame in a window
    cv2.imshow("Step 1 - Camera", frame)

    # Wait 1 ms for a key; stop if it is q
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()