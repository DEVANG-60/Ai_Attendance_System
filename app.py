from flask import Flask, render_template, Response, send_file, request, redirect
import cv2
import os
import csv
import pickle
import pandas as pd
from datetime import datetime

from capture_faces import capture_student
from train_model import train_model

app = Flask(__name__)

# ================= LIVE RECOGNITION DATA =================

latest_person = {
    "name": "Waiting...",
    "confidence": "0%",
    "time": "--:--:--",
    "status": "READY"
}

# ==========================================================
# CREATE ATTENDANCE FILE IF NOT EXISTS
# ==========================================================

if not os.path.exists("attendance.csv"):
    pd.DataFrame(
        columns=["Name", "Date", "Time"]
    ).to_csv("attendance.csv", index=False)

# ==========================================================
# FACE DETECTOR
# ==========================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("models/trainer.yml"):
    recognizer.read("models/trainer.yml")
    print("Trainer Loaded")
else:
    print("trainer.yml not found")

if os.path.exists("models/labels.pkl"):
    with open("models/labels.pkl", "rb") as file:
        names = pickle.load(file)
else:
    names = {}

# ==========================================================
# START CAMERA
# ==========================================================

camera = cv2.VideoCapture(0)

# Optional camera settings
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================================
# ATTENDANCE
# ==========================================================

last_seen = {}

def markAttendance(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Prevent duplicate within 10 seconds
    if name in last_seen:
        if (now - last_seen[name]).seconds < 10:
            return

    last_seen[name] = now

    if not os.path.exists("attendance.csv"):

        pd.DataFrame(
            columns=["Name", "Date", "Time"]
        ).to_csv("attendance.csv", index=False)

    df = pd.read_csv("attendance.csv")

    already = df[
        (df["Name"] == name) &
        (df["Date"] == date)
        ]

    if len(already) > 0:
        return

    df.loc[len(df)] = [
        name,
        date,
        time
    ]

    df.to_csv("attendance.csv", index=False)

    print(f"{name} attendance marked.")

# ==========================================================
# CAMERA STREAM
# ==========================================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (200, 200)
            )

            try:

                label, confidence = recognizer.predict(face)

                if confidence < 70:
                    name = names.get(label, "UNKNOWN")
                    color = (0,255,0)
                    markAttendance(name)
                    latest_person["name"] = name
                    latest_person["confidence"] = f"{100-int(confidence)}%"
                    latest_person["time"] = datetime.now().strftime("%H:%M:%S")
                    latest_person["status"] = "Recognized"

                else:
                    name = "UNKNOWN"
                    color = (0,0,255)
                    latest_person["name"] = "UNKNOWN"
                    latest_person["confidence"] = "0%"
                    latest_person["time"] = datetime.now().strftime("%H:%M:%S")
                    latest_person["status"] = "Unknown Face"

            except:

                name = "UNKNOWN"

                confidence = 100

                color = (0, 0, 255)

            confidence_text = max(
                0,
                min(
                    100,
                    int(100 - confidence)
                )
            )

            display = f"{name} ({confidence_text}%)"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                display,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        _, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
        )

# ==========================================================
# ROUTES
# ==========================================================
# ==========================================================
# LANDING PAGE
# ==========================================================

@app.route("/")
def landing():
    return render_template("home.html")


# ==========================================================
# LOGIN
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            return redirect("/dashboard")

        return render_template(
            "login.html",
            error="Invalid Username or Password"
        )

    return render_template("login.html")


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# ==========================================================
# DEMO PAGE
# ==========================================================

@app.route("/demo")
def demo():
    return render_template("demo.html")


# ==========================================================
# REGISTER STUDENT
# ==========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    global names

    if request.method == "POST":

        name = request.form["name"].strip()
        roll = request.form["roll"].strip()
        department = request.form["department"].strip()
        year = request.form["year"].strip()
        email = request.form["email"].strip()

        # Create students.csv
        if not os.path.exists("students.csv"):

            with open(
                    "students.csv",
                    "w",
                    newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Name",
                    "Roll",
                    "Department",
                    "Year",
                    "Email"
                ])

        # Check duplicate roll number

        duplicate = False

        with open(
                "students.csv",
                "r",
                newline=""
        ) as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if len(row) > 1 and row[1] == roll:

                    duplicate = True

                    break

        if duplicate:

            return """
            <script>
            alert("Roll Number Already Exists");
            window.location='/register';
            </script>
            """

        # Save Student

        with open(
                "students.csv",
                "a",
                newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                name,
                roll,
                department,
                year,
                email
            ])

        # Capture Faces

        capture_student(name)

        # Train AI

        train_model()

        # Reload Labels

        if os.path.exists("models/labels.pkl"):

            with open(
                    "models/labels.pkl",
                    "rb"
            ) as file:

                names = pickle.load(file)

        # Reload Model

        if os.path.exists("models/trainer.yml"):

            recognizer.read(
                "models/trainer.yml"
            )

        return """
        <script>
        alert("Student Registered Successfully");
        window.location='/dashboard';
        </script>
        """

    return render_template("register.html")


# ==========================================================
# LIVE DATA
# ==========================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/live_data")
def live_data():

    return latest_person

# ==========================================================
# LIVE ATTENDANCE TABLE
# ==========================================================

@app.route("/recent_attendance")
def recent_attendance():

    if not os.path.exists("attendance.csv"):

        return []

    df = pd.read_csv("attendance.csv")

    if len(df) == 0:

        return []

    df = df.tail(10)

    return df.iloc[::-1].to_dict(orient="records")

# ==========================================================
# STATS API
# ==========================================================

@app.route("/stats")
def stats():

    total = len([
        d for d in os.listdir("students")
        if os.path.isdir(os.path.join("students", d))
    ])

    present = 0

    if os.path.exists("attendance.csv"):

        df = pd.read_csv("attendance.csv")

        if not df.empty:

            today = datetime.now().strftime("%Y-%m-%d")

            df = df[df["Date"] == today]

            present = df["Name"].nunique()

    return {
        "total": total,
        "present": present
    }
# ==========================================================
# REPORT
# ==========================================================

@app.route("/report")
def report():

    if os.path.exists("attendance.csv"):

        df = pd.read_csv(
            "attendance.csv"
        )

        return df.to_html(
            classes="table table-striped table-bordered",
            index=False
        )

    return "<h2>No Attendance Data Found</h2>"


# ==========================================================
# DOWNLOAD CSV
# ==========================================================

@app.route("/download")
def download():

    if os.path.exists("attendance.csv"):

        return send_file(
            "attendance.csv",
            as_attachment=True
        )

    return "No attendance file found."


# ==========================================================
# ATTENDANCE CHART
# ==========================================================

@app.route("/chart")
def chart():

    if not os.path.exists(
            "attendance.csv"
    ):
        return "<h2>No Attendance Data</h2>"

    df = pd.read_csv(
        "attendance.csv"
    )

    if len(df) == 0:
        return "<h2>No Attendance Data</h2>"

    counts = df["Name"].value_counts()
    labels = counts.index.astype(str).tolist()
    values = counts.astype(int).tolist()

    return f"""
    <html>
        <head>
            <title>Attendance Analytics</title>
        </head>

    <body style="font-family:Arial;text-align:center;background:#f5f5f5;">

        <h1>Attendance Analytics</h1>

        <img src="https://quickchart.io/chart?c={{
            type:'bar',
            data:{{
            labels:{labels},
            datasets:[{{
            label:'Attendance Count',
            data:{values}
            }}]
            }}
        }}">

    </body>
</html>
"""

# ==========================================================
# STUDENT LIST
# ==========================================================
@app.route("/students")
def students():

    if not os.path.exists("students.csv"):
        return render_template(
            "students.html",
            students=[]
        )

    df = pd.read_csv("students.csv")

    return render_template(
        "students.html",
        students=df.to_dict(orient="records")
    )
# ==========================================================
# DELETE STUDENT
# ==========================================================

@app.route("/delete/<roll>")
def delete_student():

    roll = request.view_args["roll"]

    if os.path.exists("students.csv"):

        df = pd.read_csv("students.csv")

        student = df[df["Roll"].astype(str) == str(roll)]

        if len(student):

            name = student.iloc[0]["Name"]

            folder = os.path.join("students", name)

            if os.path.exists(folder):

                import shutil
                shutil.rmtree(folder)

            df = df[df["Roll"].astype(str) != str(roll)]

            df.to_csv("students.csv", index=False)

            train_model()

            if os.path.exists("models/labels.pkl"):

                global names

                with open("models/labels.pkl", "rb") as file:
                    names = pickle.load(file)

            recognizer.read("models/trainer.yml")

    return redirect("/students")


# ==========================================================
# EDIT STUDENT
# ==========================================================

@app.route("/edit/<roll>", methods=["GET", "POST"])
def edit_student():

    roll = request.view_args["roll"]

    df = pd.read_csv("students.csv")

    student = df[df["Roll"].astype(str) == str(roll)]

    if len(student) == 0:
        return "Student Not Found"

    student = student.iloc[0]

    if request.method == "POST":

        df.loc[
            df["Roll"].astype(str) == str(roll),
            ["Name", "Department", "Year", "Email"]
        ] = [

            request.form["name"],
            request.form["department"],
            request.form["year"],
            request.form["email"]

        ]

        df.to_csv("students.csv", index=False)

        return redirect("/students")

    return render_template(
        "edit_student.html",
        student=student
    )

#=====================================================================
#Analytics
#=====================================================================


@app.route("/analytics")
def analytics():

    if os.path.exists("students.csv"):
        students = pd.read_csv("students.csv")
        total = len(students)
    else:
        total = 0

    if os.path.exists("attendance.csv"):
        attendance = pd.read_csv("attendance.csv")
        present = attendance["Name"].nunique()
        counts = attendance["Name"].value_counts()
    else:
        attendance = pd.DataFrame(columns=["Name"])
        present = 0
        counts = pd.Series(dtype=int)

    absent = max(0, total - present)

    percentage = round((present / total) * 100, 2) if total else 0

    return render_template(

        "analytics.html",

        total=total,

        present=present,

        absent=absent,

        percentage=percentage,

        labels=list(counts.index),

        values=list(counts.values)

    )

# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )