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
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree_function', 'age']]

        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]  

        if prediction == 1:
            result = f"Patient is likely to have diabetes – recommend further testing. (Confidence diabetes: {proba:.1%})"
        else:
            result = f"No risk of diabetes detected. (Confidence: {(1 - proba):.1%})"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"<pre>{str(e)}</pre>"


if __name__ == '__main__':
    app.run(debug=True)
