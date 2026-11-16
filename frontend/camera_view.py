
import cv2

# Connect to the MacBook camera
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    print("Error: Cannot access camera.")
else:
    print("CATCHBot camera started successfully!")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture video.")
            break

        # Display live camera
        cv2.imshow("CATCHBot - Live Camera", frame)

        # Press Q to stop the camera
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
