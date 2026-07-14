from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
model = joblib.load("models/house_price_model.pkl")
features = joblib.load("models/features.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Create dictionary of all features initialized to 0
    input_data = {feature: 0 for feature in features}

    # Numeric inputs
    input_data["area"] = float(request.form["area"])
    input_data["bedrooms"] = int(request.form["bedrooms"])
    input_data["bathrooms"] = int(request.form["bathrooms"])
    input_data["stories"] = int(request.form["stories"])
    input_data["parking"] = int(request.form["parking"])

    # Categorical values
    categorical = {
        "mainroad": request.form["mainroad"],
        "guestroom": request.form["guestroom"],
        "basement": request.form["basement"],
        "hotwaterheating": request.form["hotwaterheating"],
        "airconditioning": request.form["airconditioning"],
        "prefarea": request.form["prefarea"],
        "furnishingstatus": request.form["furnishingstatus"],
    }

    for key, value in categorical.items():
        column = f"{key}_{value}"
        if column in input_data:
            input_data[column] = 1

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)