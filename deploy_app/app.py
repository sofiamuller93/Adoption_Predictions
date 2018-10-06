import os
import io
import numpy as np
import pandas as pd
import shutil

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K
from fuzzywuzzy import process
from fuzzywuzzy.fuzz import partial_ratio
from werkzeug.utils import secure_filename

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    global graph
    model = Xception(weights="imagenet")
    graph = K.get_session().graph

load_model()

def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        # print(request.method)
        print(request)
        print('posted')

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print('HELP!')
            file.save(filepath)

            # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            image_size = (299, 299)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():
                preds = model.predict(image)
                results = decode_predictions(preds)
                data["predictions"] = []
                print(decode_predictions(preds, top=3)[0])
                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)
                print(data)
                # indicate that the request was a success
                data["success"] = True
                predictions = [
                    {"breed": data["predictions"][0]['label'], "id": 1},
                    {"breed": data["predictions"][1]['label'], "id": 2}
                    ]

        return render_template("results.html", data=predictions)

    return render_template("index.html")


@app.route("/data/<breed>")
def unadopted(breed):

    print(breed)
    breed.replace("%20", " ")
    animals = pd.read_csv("Animals.csv")

    # FUZZY WUZZY to get breeds that are close to the return
    r = process.extractBests(breed.replace('%20', ' '), animals.Breed.to_dict(), scorer=partial_ratio, score_cutoff=70, limit=1000000000)
    
    animal_analysis = animals.loc[map(lambda x: x[-1], r)]

    from pprint import pprint
    # pprint(animal_analysis)

    animal_unadopted = animal_analysis[animal_analysis["Outcome Type"] != "Adoption"]

    pprint(animal_unadopted)

    # animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]


    # Create bins in which to place values based upon age in days
    bins = [0, 270, 450, 730, 1460, 2920, 3650, 7000]

    # Create labels for these bins
    labels = ["Less than 9 mon.", "9 to 15 mon.", "2 years", "4 years", "6 years", "8 years", "Senior"]

    age_series = pd.cut(animal_unadopted["Intake Age (days)"], bins, labels=labels, include_lowest=True)
    animal_unadopted["age_range"] = age_series
    animal_unadopted.dropna()
    print("did i make here")

    age_df = animal_unadopted.groupby("age_range").count()["Breed"]
    labels = age_df.index.tolist()
    counts = age_df.values.tolist()
    data = {"x": labels, "y": counts}
    return jsonify(data)

@app.route("/gender/<breed>")
def gender(breed):
    breed.replace("%20", " ")
    animals = pd.read_csv("Animals.csv")

    r = process.extractBests(breed.replace('%20', ' '), animals.Breed.to_dict(), scorer=partial_ratio, score_cutoff=70,
                             limit=1000000000)

    animal_analysis = animals.loc[map(lambda x: x[-1], r)]


    animal_unadopted = animal_analysis[animal_analysis["Outcome Type"] != "Adoption"]
    # animal_unadopted = animal_analysis[animal_unadopted["Breed"] == breed]

    df = animal_unadopted.groupby("Gender").count()["Breed"]
    labels = df.index.tolist()
    counts = df.values.tolist()
    data = {"x": labels, "y": counts}
    return jsonify(data)


@app.route("/index.html")
def index():
    return render_template('index.html')

@app.route("/adoption_statistics.html")
def adoption_statistics():
    return render_template('adoption_statistics.html')

@app.route("/cat_stats.html")
def cat_stats():
    return render_template('cat_stats.html')

@app.route("/dog_stats.html")
def dog_stats():
    return render_template('dog_stats.html')

@app.route("/theteam.html")
def theteam():
    return render_template('theteam.html')

@app.route("/resources.html")
def resources():
    return render_template('resources.html')

@app.route("/machine_learning.html")
def machine_learning():
    return render_template('machine_learning.html')

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=80)
    app.run(debug=True)