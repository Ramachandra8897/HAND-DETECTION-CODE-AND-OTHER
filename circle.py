import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw circles on detected faces
    for (x, y, w, h) in faces:
        # Calculate the center and radius for the circle
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2  # Approximate radius

        # Draw the circle
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Face Detection with Circle', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
