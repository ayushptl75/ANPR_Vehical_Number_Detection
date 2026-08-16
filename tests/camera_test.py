import cv2

if __name__ == "__main__":
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()