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

# These models are only for demonstration purposes and will serve "random" values
# You might as well use pretrained models from Keras - check https://keras.io/applications/
# Make sure to give your models meaningful names 
MODEL_NAMES = ["test-model", "test-model", "test-model"]
AMOUNT_OF_MODELS = len(MODEL_NAMES)
MODELS = []
GRAPHS = []
SESSIONS = []

def load_models():
    for i in range(AMOUNT_OF_MODELS):
        load_single_model("models/" + MODEL_NAMES[i] + ".h5")
        print("\nModel ", str(i+1), " of ",  AMOUNT_OF_MODELS, " loaded.")
    print('Ready to go! Visit -> http://127.0.0.1:5000/')   

# Thanks to all participants who contributed at https://github.com/keras-team/keras/issues/8538
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


def models_predict(file_path):
    # Beware to adapt the preprocessing to the input of your trained models
    img = image.load_img(file_path, target_size=(64, 64))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = []

    for i in range(AMOUNT_OF_MODELS):
        with GRAPHS[i].as_default():
            with SESSIONS[i].as_default():
                numeric_prediction = MODELS[i].predict(x)

                # Adapt the prediction format to your models output
                binary_prediction = 0

                if numeric_prediction > 0.5:
                    binary_prediction = 1

                preds.append(binary_prediction)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the image from the POST request
        image_to_predict = request.files['image']

        # Save the image to ./uploads
        basepath = os.path.dirname(__file__)
        local_image_path = os.path.join(
            basepath, 'uploads', secure_filename(image_to_predict.filename))
        image_to_predict.save(local_image_path)

        predictions = models_predict(local_image_path)
        # Remove image after prediction is done
        os.remove(local_image_path)

        return jsonify(predictions)   


if __name__ == '__main__':
    load_models()
    wsgi_server = WSGIServer(('0.0.0.0', 5000), app)
    wsgi_server.serve_forever()
