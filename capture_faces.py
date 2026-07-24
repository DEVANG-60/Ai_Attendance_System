import cv2
import os


def capture_student(name, image_count=100):
    """
    Capture face images for a new student.
    Images are stored in:
    students/<Student_Name>/
    """

    save_path = os.path.join("students", name.replace(" ", "_"))

    os.makedirs(save_path, exist_ok=True)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(0)

    count = 0

    print("\nLook at the camera...")
    print("Capturing images...\n")

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (200, 200))

            filename = os.path.join(save_path, f"{count}.jpg")

            cv2.imwrite(filename, face)

            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(
                frame,
                f"{count}/{image_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        cv2.imshow("Student Registration", frame)

        if cv2.waitKey(1) == 27:
            break

        if count >= image_count:
            break

    camera.release()

    cv2.destroyAllWindows()

    print(f"\n{name} Registered Successfully")