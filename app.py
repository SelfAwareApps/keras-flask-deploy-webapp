import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, model_from_json
from keras.preprocessing import image

# Tensorflow
from tensorflow import Graph, Session

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__, template_folder='frontend', static_folder='frontend', static_url_path='')
CORS(app)

MODEL_NAMES = ["A", "B", "C"]
AMOUNT_OF_MODELS = len(MODEL_NAMES)
MODELS = []
GRAPHS = []
SESSIONS = []

def load_models():
    for i in range(AMOUNT_OF_MODELS):
        load_single_model("models/" + MODEL_NAMES[i] + ".h5")
        print("Model " + str(i) + " of "+ AMOUNT_OF_MODELS + " loaded.")
    print('Ready to go! Visit -> http://127.0.0.1:5000/')   


def load_single_model(path):
    graph = Graph()
    with graph.as_default():
        session = Session()
        with session.as_default():
            model = load_model(path)
            model._make_predict_function() 
            
            MODELS.append(model)
            GRAPHS.append(graph)
            SESSIONS.append(session)


def models_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = []

    for i in range(AMOUNT_OF_MODELS):
        with GRAPHS[i].as_default():
            with SESSIONS[i].as_default():
                numeric_prediction = MODELS[i].predict(x)
                binary_prediction = 0

                if numeric_prediction > 0.5:
                    binary_prediction = 1

                preds.append(binary_prediction)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)

        preds = models_predict(img_path)
        os.remove(img_path)

        return jsonify(preds)   


if __name__ == '__main__':
    load_models()
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
