<<<<<<< HEAD
import base64
import argparse
import base64
import json
from io import BytesIO
import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
import math
import cv2

from keras.models import model_from_json

tf.python.control_flow_ops = tf

STEERING_ADJUSTMENT = 1

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .2
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 8

def return_image(img, color_change=True):
    # Take out the dash and horizon
    img_shape = img.shape
    img = img[60:img_shape[0] - 25, 0:img_shape[1]]
    # assert crop_img.shape[0] == IMAGE_HEIGHT_CROP
    # assert crop_img.shape[1] == IMAGE_WIDTH_CROP
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))
    return np.float32(img)

@sio.on('telemetry')
def telemetry(sid, data):
    global old_steering_angle
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]

    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = np.array(image)
    new_size_col,new_size_row = 64, 64
    shape = image.shape
    image = image[math.floor(shape[0]/5.):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),         interpolation=cv2.INTER_AREA)
    image = image/255.

    transformed_image_array = image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(saved_model.predict(transformed_image_array, batch_size=1) * STEERING_ADJUSTMENT)

    # The driving model currently just outputs a constant throttle. Feel free to edit this.

    if abs(steering_angle) > .07:
        throttle = .05
    else:
        throttle = AUTONOMOUS_THROTTLE

    print('Angle: {0}, Throttle: {1}'.format(round(steering_angle, 4), throttle))

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    import model

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json.')
    args = parser.parse_args()

    saved_model = None
    print (args.model)
    with open(args.model, 'r') as jfile:
        saved_model = model_from_json(json.load(jfile))
   
    saved_model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    saved_model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
=======
import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
from model import preprocess_image


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    speed = float(speed)
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # Add the preprocessing step
    image_array = preprocess_image(image_array)

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if speed < 10.0:
        throttle = 0.7
    elif speed < 15.0:
        throttle = 0.3
    elif speed < 22.0:
        throttle = 0.18
    else:
        throttle = 0.15

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # model = model_from_json(json.loads(jfile.read()))
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
>>>>>>> 9d5a6cbfb4067c68320b3a44b4012bc59ac2d37f
