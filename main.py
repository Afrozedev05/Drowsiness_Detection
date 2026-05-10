import cv2
import time
import winsound

# Load classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Start camera
cap = cv2.VideoCapture(0)

# Variables
eyes_closed_start = None
no_face_counter = 0
DROWSY_TIME = 3  # seconds
NO_FACE_TEXT = "NO DRIVER DETECTED"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    eyes_detected = False
    face_detected = False

    # If no face detected
    if len(faces) == 0:

        no_face_counter += 1

        # Only show no driver after several missed frames
        if no_face_counter > 15:

            eyes_closed_start = None

            cv2.putText(
                frame,
                NO_FACE_TEXT,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

    else:

        no_face_counter = 0
        face_detected = True

        for (x, y, w, h) in faces:

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=12,
                minSize=(30, 30)
            )

            # Eyes detected
            if len(eyes) > 0:

                eyes_detected = True

                for (ex, ey, ew, eh) in eyes:

                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex+ew, ey+eh),
                        (0, 255, 0),
                        2
                    )

    # Drowsiness logic
    if face_detected:

        if not eyes_detected:

            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            elapsed = time.time() - eyes_closed_start

            cv2.putText(
                frame,
                f"Eyes Closed: {int(elapsed)} sec",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Trigger alert only after long duration
            if elapsed >= DROWSY_TIME:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                winsound.Beep(1000, 500)

        else:
            eyes_closed_start = None

            cv2.putText(
                frame,
                "DRIVER ACTIVE",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("AI Driver Drowsiness Detection", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()