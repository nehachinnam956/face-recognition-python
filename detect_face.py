import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(r"C:\Users\nehab\OneDrive\Desktop\FaceRecognitionProject\haarcascade_frontalface_default.xml")

def detect_faces_in_image(image_path):
    """Detect faces in an image and display the result."""
    img = cv2.imread(image_path.strip('"'))  # Remove extra quotes from path

    if img is None:
        print("Error: Image not found! Check the file path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_video():
    """Detect faces in live webcam feed."""
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Cannot access webcam!")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Live Face Detection", img)

        # Press 'Esc' key (27) to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution flow
choice = input("Enter 'image' to detect faces in an image or 'video' for live webcam detection: ").strip().lower()

if choice == "image":
    img_path = input("Enter the full path of the image: ").strip()
    detect_faces_in_image(img_path)
elif choice == "video":
    detect_faces_in_video()
else:
    print("Invalid input! Please enter 'image' or 'video'.")



