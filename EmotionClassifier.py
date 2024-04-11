# Imports

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import cv2

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

# Set up face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Set up labels
label_to_text = {0: "anger",
                    1: "disgust",
                    2: "fear",
                    3: "happiness",
                    4: "sadness",
                    5: "surprise",
                    6: "neutral"}

# Obtain best model
file_name = 'best_model.keras'
checkpoint_path = os.path.join('checkpoints', file_name) 
best_model = tf.keras.models.load_model(checkpoint_path)

# Access webcam
video_capture = cv2.VideoCapture(0)

# Function to classify emotions in video stream
def detect_classify_emotion(vid):
    # convert to grayscale for processing
    img_gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    # detect faces available in video stream
    faces = face_classifier.detectMultiScale(img_gray, 1.1, 5, minSize = (40, 40))

    # process each available face in video stream
    for (x, y, w, h) in faces:
        # draw bounding box around face
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # pre-process face for model prediction
        face = img_gray[y:y + h, x:x + w]
        cropped_face = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)

        # preduct emotion using best model
        predicted_emotion = best_model.predict(cropped_face)
        max_index = int(np.argmax(predicted_emotion))
        cv2.putText(vid, label_to_text[max_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return faces

# Use a loop for real-time face detection
while True:

    result, video_frame = video_capture.read() # read frames from video
    if result is False:
        break # terminate the loop if the frame is not read successfully

    faces = detect_classify_emotion(video_frame)

    cv2.imshow("Dylan's Emotion Classifier Project", video_frame) # display processed frame in window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()