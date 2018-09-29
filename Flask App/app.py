import os
import io
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K

<<<<<<< HEAD
from flask import Flask, request, redirect, url_for, jsonify
=======
from flask import Flask, request, redirect, url_for, jsonify, render_template
>>>>>>> 26a72476867cee0aff4ccb44cf5e95b56de8ae37

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None


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
<<<<<<< HEAD
=======
        print(request.method)
        print('posted')
>>>>>>> 26a72476867cee0aff4ccb44cf5e95b56de8ae37
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

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

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

        return jsonify(data)

<<<<<<< HEAD
    return render_template("index.html")
    


if __name__ == "__main__":
    app.run(debug=True)
=======
    return render_template("index.html", dict = jsonify(data))

@app.route("/resources")
def index():
    """Return the homepage."""
    return render_template("resources_intakes.html")

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 26a72476867cee0aff4ccb44cf5e95b56de8ae37
