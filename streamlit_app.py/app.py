from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# Loading models
try:
    with open('dtr.pkl', 'rb') as f:
        dtr = pickle.load(f)
    with open('preprocesser.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model or preprocessor: {e}")
    dtr, preprocessor = None, None

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']
        except ValueError as e:
            return render_template('index.html', prediction='Error in input data')

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        try:
            transformed_features = preprocessor.transform(features)
            prediction = dtr.predict(transformed_features)
            prediction_value = prediction[0]
        except Exception as e:
            return render_template('index.html', prediction=f'Error in prediction: {e}')

        return render_template('index.html', prediction=prediction_value)

if __name__ == "__main__":
    app.run(debug=True)
