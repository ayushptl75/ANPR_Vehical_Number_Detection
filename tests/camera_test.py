import cv2

# 0 = Default laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame")
        break

    cv2.imshow("Laptop Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()