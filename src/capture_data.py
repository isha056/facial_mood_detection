import cv2
import os
import time

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def capture_images(label, num_samples=100):
    """
    Captures images from the webcam for a specific emotion label.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create directory for the label
    label_dir = os.path.join(DATA_DIR, 'raw', label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f"Starting capture for label: {label}")
    print("Press 's' to start capturing/resume, 'q' to quit this label.")
    
    count = 0
    capturing = False
    
    # Load Haar cascade for face detection to only save face regions (optional but good)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle and helpful text
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Label: {label}, Count: {count}/{num_samples}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if capturing:
            cv2.putText(display_frame, "CAPTURING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If a face is found, save it
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Save the face region
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                
                img_path = os.path.join(label_dir, f"{label}_{int(time.time()*1000)}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                time.sleep(0.1) # small delay
                
        else:
            cv2.putText(display_frame, "Press 's' to start", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capture Data', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            capturing = not capturing

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing for {label}. Total: {count}")

if __name__ == "__main__":
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise']
    print("Welcome to the Facial Mood Dataset Creator!")
    print("We will capture images for the following emotions:", emotions)
    
    for emotion in emotions:
        choice = input(f"Do you want to capture images for '{emotion}'? (y/n): ")
        if choice.lower() == 'y':
            samples = int(input(f"How many samples for {emotion}? (default 100): ") or 100)
            capture_images(emotion, samples)
    
    print("Data collection complete.")
