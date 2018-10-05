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
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#################################################
# Flask Setup
#################################################
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


@app.route('/', methods=['GET'])
def index():
    return render_template('test.html', title='Upload files')


@app.route('/upload_file', methods=['POST']) 
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

        animal_outcome = pd.read_csv('Animals.csv')

        r = process.extractBests(breed.replace('%20', ' '), animal_outcome.Breed.to_dict(), scorer=partial_ratio, score_cutoff=70, limit=1000000000)
        animal_analysis = animal_outcome.loc[map(lambda x: x[-1], r)]
        animal_analysis["Prediction_Breed"] = prediction[0][1]
        animal_analysis["Prediction_ID"] = prediction[0][0]
        # json_data = animal_analysis.to_json(orient='records')
        json_data = animal_analysis.to_json()
        return json_data
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




@app.route("/data/<json_data>")
def unadopted(json_data):
    #breed.replace("%20", " ")
    animals = pd.read_json(json_data)
    from pprint import pprint
    pprint(animals)
    
    animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
    # animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

    # Create bins in which to place values based upon age in days
    bins = [0, 270, 450, 730, 1460, 2920, 3650, 7000]

    # Create labels for these bins
    labels = ["Less than 9 mon.", "9 to 15 mon.", "2 years", "4 years", "6 years", "8 years", "Senior"]

    age_series = pd.cut(animal_unadopted["Intake Age (days)"], bins, labels=labels, include_lowest=True)
    animal_unadopted["age_range"] = age_series
    animal_unadopted.dropna()

    age_df = animal_unadopted.groupby("age_range").count()["Breed"]
    labels = age_df.index.tolist()
    counts = age_df.values.tolist()
    data = {"x": labels, "y": counts}
    return jsonify(data)

@app.route("/age/json_data")
def adopt_by_age(json_data):

    json_data.replace("%20", " ")
    # print(breed)
    animals = pd.read_json(json_data)

    animal_unadopted = animals[animals["Outcome Type"]!="Adoption"]
    age_unadopted = animal_unadopted[animal_unadopted["Breed"]].groupby("Intake Age (days)").count()
    age_unadopted["Breed"] = age_unadopted["Breed"]/age_unadopted["Breed"].sum()
    
    animal_adopted = animals[animals["Outcome Type"]=="Adoption"]
    age_adopted = animal_adopted[animal_adopted["Breed"]].groupby("Intake Age (days)").count()
    age_adopted["Breed"] = age_adopted["Breed"]/age_adopted["Breed"].sum()
    age_adopted["Breed"].sum()


    delta = age_unadopted["Breed"] - age_adopted["Breed"]
    delta = delta.fillna(0)
    x = delta.index.values.tolist()
    y = delta.values.tolist()
    
    data = {
        "x": x,
        "y": y,
        "title": "Age vs Adoptability"
    }

    return jsonify(data)


@app.route("/gender/json_data")
def gender(json_data):
    animals = pd.read_json(json_data)
    animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
    # animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

    df = animal_unadopted.groupby("Gender").count()["Breed"]
    labels = df.index.tolist()
    counts = df.values.tolist()
    data = {"x": labels, "y": counts}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)