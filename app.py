import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, Response
import cv2
from mtcnn import MTCNN
from keras.models import load_model

SIZE=224
#load the model
VGG16_model=load_model('model_cropped.h5')

#create the app
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def crop_face_and_return(image):
    cropped_face = None
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if faces:
        x, y, width, height = faces[0]['box']

        cropped_face = image[y:y + height, x:x + width]

    return cropped_face


def get_max_indx(arr):
    return arr.index(max(arr))

def generate_frames():
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

            #select the image from the video
            cv2.imwrite('/image' + '.jpg', frame)
            image = cv2.imread('/image' + '.jpg')
            cropped_face = crop_face_and_return(image)
            if cropped_face is not None:
                # Convert the NumPy array 'cropped_face' into a PIL Image
                pil_image = Image.fromarray(cropped_face, 'RGB')

                pil_image = pil_image.resize((SIZE, SIZE))

                cropped_face = np.array(pil_image)

                image = tf.reshape(cropped_face, (1, SIZE, SIZE, 3))

            #output the prediction text
            prediction = VGG16_model.predict(image)
            # pred = get_max_indx(prediction)
            print(prediction)
            # if prediction == 1:
            #     render_template("index1.html", prediction_text="Heart Disease detected")
            # else:
            #     render_template("index1.html", prediction_text="No Heart Disease")

            #display the video
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
