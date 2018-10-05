from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
# import tensorflow as tf
import pandas as pd
import keras
from keras.preprocessing import image
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)

from keras import backend as K

from fuzzywuzzy import process
from fuzzywuzzy.fuzz import partial_ratio


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Images'

model = None
graph = None


def load_model():
    global model
    global graph
    model = Xception(
        include_top=True,
        weights='imagenet')
    graph = K.get_session().graph


load_model()


@app.route('/')
def index ():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST','GET']) 
def upload_file ():
    data = {"success": False}
    print(request.files)
    if request.files.get('file'):
        # read the file
        file = request.files['file']

        # read the filename
        filename = file.filename

        # create a path to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        prediction = predict(filepath)
        breed = prediction[0][1]

        animal_outcome = pd.read_csv('../Animals.csv')

        r = process.extractBests(breed.replace('_', ' '), animal_outcome.Breed.to_dict(), scorer=partial_ratio, score_cutoff=70, limit=1000000000)
        animal_analysis = animal_outcome.loc[map(lambda x: x[-1], r)]
        return animal_analysis.to_json(orient='records')
        # return render_template('table.html', dogs=animal_analysis.to_dict(orient='records')) 

    return jsonify({"error": "there is an error!"})



def predict(image_path):
    """Use Xception to label image"""
    global graph
    with graph.as_default():
        image_size = (299, 299)
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        print('Predicted:', decode_predictions(predictions, top=1)[0])
        return decode_predictions(predictions, top=1)[0]


if __name__ == "__main__":
    app.run(debug=True)