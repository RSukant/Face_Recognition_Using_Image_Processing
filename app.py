from flask import Flask, render_template, Response, request, redirect, url_for
from flask_mysqldb import MySQL
import os
import cv2
import face_recognition
import numpy as np
from base64 import b64decode, b64encode
import dlib

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


app = Flask(__name__)


# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'MASTER_sukant170504'
app.config['MYSQL_DB'] = 'face_images'
mysql = MySQL(app)

# Initialize camera
camera = cv2.VideoCapture(0)

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to convert image to base64 string
def convert_to_base64(file_path):
    with open(file_path, "rb") as img_file:
        return b64encode(img_file.read()).decode('utf-8')


# Function to save image to database
def save_image_to_db(image_name, image_data):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO images (image_name, image_data) VALUES (%s, %s)", (image_name, image_data))
    mysql.connection.commit()
    cur.close()


# Function to retrieve image from database
def get_image_from_db():
    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute("SELECT image_name, image_data FROM images")
        rows = cur.fetchall()
        images = {}
        for row in rows:
            image_name, image_data = row
            images[image_name] = b64decode(image_data)  # Decode base64 string to bytes
        cur.close()
        return images


# Assign images from database to variables
def assign_images_from_db(images):
    known_face_encodings = []
    known_face_names = []
    for image_name, image_data in images.items():
        img_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(image_name)
    return known_face_encodings, known_face_names


# Load known face encodings and names from the database
images_from_db = get_image_from_db()
known_face_encodings, known_face_names = assign_images_from_db(images_from_db)


def detect_face_dlib(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Extract face locations as tuples of (top, right, bottom, left)
    face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in faces]

    return face_locations



def encode_faces_dlib(frame, face_landmarks):
    # Convert frame to RGB (Dlib uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract face encodings using facial landmarks
    face_encodings = [np.array(face_descriptor) for face_descriptor in
                      face_recognition.face_encodings(rgb_frame, face_landmarks)]

    return face_encodings



import numpy as np

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces using Dlib
                faces = detector(gray)

                # Extract face locations as tuples of (top, right, bottom, left)
                face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in faces]

                # Convert frame to RGB (Dlib uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract face encodings using facial landmarks
                face_encodings = [np.array(face_descriptor) for face_descriptor in
                                  face_recognition.face_encodings(rgb_frame, face_locations)]

                # Match known faces
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    if any(matches):
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] < 0.99:
                            # Calculate percentage match
                            match_percentage = (1 - face_distances[best_match_index]) * 100
                            name = f"{known_face_names[best_match_index]} ({match_percentage:.2f}%)"
                    face_names.append(name)

                # Draw rectangles and labels on the frame
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Set highlighting color (e.g., green)
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)

                # Convert frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Yield frame for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print("Error processing frame:", e)




@app.route('/')
def index():
    return render_template('home.html')


@app.route('/index')
def index_page():
    return render_template('index.html')


@app.route('/upload')
def render_upload_page():
    return render_template('upload.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_page():
    if 'image' in request.files:
        image = request.files['image']
        image_name = request.form['image_name']  # Get image name from form
        image.save(os.path.join('uploads', image.filename))
        image_path = os.path.join('uploads', image.filename)
        image_data = convert_to_base64(image_path)
        save_image_to_db(image_name, image_data)  # Pass image name to the function

        # Add the newly uploaded image to the list of known face encodings and names
        img_array = np.frombuffer(b64decode(image_data), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(image_name)

        os.remove(image_path)  # Remove the file after saving to database
        return redirect(url_for('index_page'))
    return "No image provided"


@app.route('/display')
def display():
    image_data = get_image_from_db()
    if image_data:
        image_name, image_data = image_data
        return f"<h2>{image_name}</h2><img src='data:image/jpeg;base64,{image_data}' alt='{image_name}'>"
    return "No image found in database"

@app.route('/home')
def home_page():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
