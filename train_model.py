import cv2
import os
import numpy as np
import pickle

# ==========================
# Train Face Recognition Model
# ==========================

def train_model():

    # Face detector
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Create recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    label_map = {}
    current_label = 0

    students_folder = "students"

    # Check if students folder exists
    if not os.path.exists(students_folder):
        print("Students folder not found.")
        return

    print("\nScanning Student Images...\n")

    # Read every student's folder
    for student in os.listdir(students_folder):

        student_path = os.path.join(students_folder, student)

        if not os.path.isdir(student_path):
            continue

        label_map[current_label] = student

        print(f"Processing : {student}")

        for image_name in os.listdir(student_path):

            image_path = os.path.join(student_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            detected = face_detector.detectMultiScale(
                img,
                scaleFactor=1.3,
                minNeighbors=5
            )

            # If face detected crop it
            if len(detected) > 0:

                x, y, w, h = detected[0]

                face = img[y:y+h, x:x+w]

                face = cv2.resize(face, (200, 200))

            else:

                # Already cropped image
                face = cv2.resize(img, (200, 200))

            faces.append(face)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("\nNo Images Found.")
        return

    print("\nTraining Model...\n")

    recognizer.train(faces, np.array(labels))

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save trained model
    recognizer.save("models/trainer.yml")

    # Save label dictionary
    with open("models/labels.pkl", "wb") as file:
        pickle.dump(label_map, file)

    print("\n==============================")
    print(" Model Trained Successfully ")
    print("==============================")
    print(f"Students Trained : {len(label_map)}")
    print(f"Images Used      : {len(faces)}")
    print("Model Saved      : models/trainer.yml")
    print("Labels Saved     : models/labels.pkl")
    print("==============================\n")


# ==========================
# Run Directly
# ==========================
if __name__ == "__main__":
    train_model()