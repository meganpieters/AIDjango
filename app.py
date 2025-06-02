from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictor')
def predictor():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[x]) for x in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            result = "Patient is likely to have diabetes – recommend further testing."
        else:
            result = "No risk of diabetes detected."
        return render_template('index.html', prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
