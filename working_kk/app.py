# # Dependencies
# import matplotlib.pyplot as plt
# %matplotlib inline

# import os
# import numpy as np
# import tensorflow as tf

# import keras
# from keras.preprocessing import image
# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions)
# from flask import Flask, jsonify, render_template
# import pandas as pd

# #################################################
# # Flask Setup
# #################################################
# app = Flask(__name__)


# @app.route("/")
# # Refactor above steps into reusable function
# def predict(image_path):
#     """Use Xception to label image"""
#     img = image.load_img(image_path, target_size=image_size)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     predictions = model.predict(x)
#     plt.imshow(img)
#     print('Predicted:', decode_predictions(predictions, top=3)[0])

# def index():

#     predictions = [
#         {"breed": "Pit Bull Mix", "id": 1},
#         {"breed": "Domestic Shorthair Mix", "id": 2}
#     ]
#     return render_template('index.html', data=predictions)


# # @app.route("/data/<breed>")
# # def unadopted(breed):
# #     breed.replace("%20", " ")
# #     animals = pd.read_csv("Animals.csv")
# #     animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
# #     animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

# #     # Create bins in which to place values based upon age in days
# #     bins = [0, 270, 450, 730, 1460, 2920, 3650, 7000]

# #     # Create labels for these bins
# #     labels = ["Less than 9 mon.", "9 to 15 mon.", "2 years", "4 years", "6 years", "8 years", "Senior"]

# #     age_series = pd.cut(animal_unadopted["Intake Age (days)"], bins, labels=labels, include_lowest=True)
# #     animal_unadopted["age_range"] = age_series
# #     animal_unadopted.dropna()

# #     age_df = animal_unadopted.groupby("age_range").count()["Breed"]
# #     labels = age_df.index.tolist()
# #     counts = age_df.values.tolist()
# #     data = {"x": labels, "y": counts}
# #     return jsonify(data)


# # @app.route("/age/<breed>")
# # def adopt_by_age(breed):

# #     breed.replace("%20", " ")
# #     print(breed)
# #     animals = pd.read_csv("Animals.csv")

# #     animal_unadopted = animals[animals["Outcome Type"]!="Adoption"]
# #     age_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed].groupby("Intake Age (days)").count()
# #     age_unadopted["Breed"] = age_unadopted["Breed"]/age_unadopted["Breed"].sum()
    
# #     animal_adopted = animals[animals["Outcome Type"]=="Adoption"]
# #     age_adopted = animal_adopted[animal_adopted["Breed"] == breed].groupby("Intake Age (days)").count()
# #     age_adopted["Breed"] = age_adopted["Breed"]/age_adopted["Breed"].sum()
# #     age_adopted["Breed"].sum()


# #     delta = age_unadopted["Breed"] - age_adopted["Breed"]
# #     delta = delta.fillna(0)
# #     x = delta.index.values.tolist()
# #     y = delta.values.tolist()
    
# #     data = {
# #         "x": x,
# #         "y": y,
# #         "title": "Age vs Adoptability"
# #     }

# #     return jsonify(data)


# # @app.route("/gender/<breed>")
# # def gender(breed):
# #     breed.replace("%20", " ")
# #     animals = pd.read_csv("Animals.csv")
# #     animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
# #     animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

# #     df = animal_unadopted.groupby("Gender").count()["Breed"]
# #     labels = df.index.tolist()
# #     counts = df.values.tolist()
# #     data = {"x": labels, "y": counts}
# #     return jsonify(data)

# if __name__ == "__main__":
#     app.run(debug=True)

import os
import io
import numpy as np
import pandas as pd

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
    model = Xception(include_top=True, weights="imagenet")
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
        print(request)
        
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("HELP!")
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

                # indicate that the request was a success
                data["success"] = True
                print(data)
                # {'success': True, 'predictions': [{'label': 'redbone', 'probability': 0.6793243885040283}, {'label': 'golden_retriever', 'probability': 0.016980241984128952}, {'label': 'black-and-tan_coonhound', 'probability': 0.015220019035041332}, {'label': 'basset', 'probability': 0.012955794110894203}, {'label': 'Chesapeake_Bay_retriever', 'probability': 0.012856719084084034}]}
                # [('n02090379', 'redbone', 0.6793244), ('n02099601', 'golden_retriever', 0.016980242), ('n02089078', 'black-and-tan_coonhound', 0.015220019)]
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

if __name__ == "__main__":
    app.run(debug=True)
