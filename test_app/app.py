from flask import Flask, jsonify, render_template
import pandas as pd

#################################################
# Flask Setup
#################################################
app = Flask(__name__)


@app.route("/")
def index():

    predictions = [
        {"breed": "Pit Bull Mix", "id": 1},
        {"breed": "Domestic Shorthair Mix", "id": 2}
    ]
    return render_template('index.html', data=predictions)


@app.route("/data/<breed>")
def unadopted(breed):
    breed.replace("%20", " ")
    animals = pd.read_csv("Animals.csv")
    animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
    animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

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


@app.route("/age/<breed>")
def adopt_by_age(breed):

    breed.replace("%20", " ")
    print(breed)
    animals = pd.read_csv("Animals.csv")

    animal_unadopted = animals[animals["Outcome Type"]!="Adoption"]
    age_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed].groupby("Intake Age (days)").count()
    age_unadopted["Breed"] = age_unadopted["Breed"]/age_unadopted["Breed"].sum()
    
    animal_adopted = animals[animals["Outcome Type"]=="Adoption"]
    age_adopted = animal_adopted[animal_adopted["Breed"] == breed].groupby("Intake Age (days)").count()
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


@app.route("/gender/<breed>")
def gender(breed):
    breed.replace("%20", " ")
    animals = pd.read_csv("Animals.csv")
    animal_unadopted = animals[animals["Outcome Type"] != "Adoption"]
    animal_unadopted = animal_unadopted[animal_unadopted["Breed"] == breed]

    df = animal_unadopted.groupby("Gender").count()["Breed"]
    labels = df.index.tolist()
    counts = df.values.tolist()
    data = {"x": labels, "y": counts}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)