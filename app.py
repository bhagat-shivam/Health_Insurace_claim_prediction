from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs
        age = float(request.form['age'])
        sex = 1 if request.form['sex'] == 'male' else 0
        weight = float(request.form['weight'])
        bmi = float(request.form['bmi'])
        hereditary_map = {
            'NoDisease': 0, 'Epilepsy': 1, 'EyeDisease': 2, 'Alzheimer': 3,
            'Arthritis': 4, 'HeartDisease': 5, 'Diabetes': 6, 'Cancer': 7, 'High BP': 8, 'Obesity': 9
        }
        hereditary = hereditary_map[request.form['hereditary_diseases']]
        dependents = int(request.form['no_of_dependents'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        city = int(request.form['city'])
        bp = int(request.form['bloodpressure'])
        diabetes = 1 if request.form['diabetes'] == 'yes' else 0
        regular_ex = 1 if request.form['regular_ex'] == 'yes' else 0
        job_title = int(request.form['job_title'])

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            age, sex, weight, bmi, hereditary, dependents, smoker,
            city, bp, diabetes, regular_ex, job_title
        ]], columns=['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents',
                     'smoker', 'city', 'bloodpressure', 'diabetes', 'regular_ex', 'job_title'])

        # Make prediction
        prediction = rf_model.predict(input_data)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
