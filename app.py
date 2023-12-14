import time
from math import inf

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, Response, flash, jsonify, send_from_directory
import cv2
from mtcnn import MTCNN
from keras.models import load_model

SIZE = 224
# load the model
VGG16_model = load_model('yarab.h5')

# create the app
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def crop_face_and_return(image):
    cropped_face = None
    detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(image, 1.1, 7)
    for (x, y, w, h) in faces:
        cropped_face = image[y:y + h, x:x + w]
    return cropped_face


# Function to check if eyes are closed based on aspect ratio
def are_eyes_closed(eyes):
    awake = 0
    for eye in eyes:
        (x, y, w, h) = eye
        aspect_ratio = float(w) / h  # the greater the value the more sleepy
        # Set a threshold for the aspect ratio to determine closed eyes
        closed_threshold = 5.0  # You may need to adjust this threshold for your specific case
        if aspect_ratio < closed_threshold:
            awake += 1
    if awake > 0:
        return False
    else:
        return True


prediction = []


def generate_frames():
    timey = 0
    i = 0
    last_pred = 0
    while True:
        i += 1
        ## read the camera frame
        success, frame = camera.read()
        if time.time() - timey > 5:
            timey = time.time()
            if not success:
                break
            else:
                detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
                faces = detector.detectMultiScale(frame, 1.1, 7)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # roi_color = None
                # roi_gray = None

                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                # select the frame from the video
                cropped_face = crop_face_and_return(gray)
                if cropped_face is not None:
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)
                    # Convert the NumPy array 'cropped_face' into a PIL Image
                    pil_image = Image.fromarray(cropped_face, 'RGB')

                    pil_image = pil_image.resize((SIZE, SIZE))

                    cropped_face = np.array(pil_image)

                    image = tf.reshape(cropped_face, (1, SIZE, SIZE, 3))

                    # output the prediction text
                    pred = VGG16_model.predict(image)
                    pred = np.argmax(pred)
                    print(pred)
                    print("last pred",last_pred)
                    if pred == 1 :
                        if last_pred == 1:
                            prediction.append(1)
                    else:
                        prediction.append(pred)
                    last_pred = pred


                else:
                    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #
                    # # Perform eye detection
                    # eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors= 15)
                    if len(eyes) == 0:
                        prediction.append(-1)
                    elif are_eyes_closed(eyes):
                        prediction.append(1)
                    else:
                        prediction.append(0)
                print(prediction)

            # display the video
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/_stuff', methods=['GET'])
def stuff():
    message = ''
    if len(prediction):
        if len(prediction) % 5 == 0: # each 5 readings of the prediction
            arr = prediction[-5:]
            message = 'the average arr'
        else:
            l_pred = prediction[-1]
            if l_pred == 0:
                message = 'active'
            elif l_pred == 2:
                message = 'yawn'
            elif l_pred == -1:
                message = 'absent'
            elif l_pred == 3:
                message = 'sleep eye'
            elif l_pred == 4:
                message = 'active eye'
            else:
                message = 'sleep'
    return jsonify(result=message)


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ensure that static files are served
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)
