from flask import Flask, render_template, request
import joblib
import numpy as np

# Load your trained model
model = joblib.load('Final_HD_Prediction_Model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    percentage_probability = ""
    risk_message = ""

    if request.method == 'POST':
        # Directly retrieve and convert feature values from the form
        bmi = request.form.get('bmi', type=float)
        physical_health = request.form.get('physical_health', type=float)
        mental_health = request.form.get('mental_health', type=float)
        sleep_time = request.form.get('sleep_time', type=float)
        smoking = request.form.get('smoking', type=int)  # 0 for No, 1 for Yes
        stroke = request.form.get('stroke', type=int)    # 0 for No, 1 for Yes
        diff_walking = request.form.get('diff_walking', type=int)  # 0 for No, 1 for Yes
        sex = request.form.get('sex', type=int)          # 0 for Female, 1 for Male
        age_category = request.form.get('age_category', type=int)
        gen_health = request.form.get('gen_health', type=int)

        # Combine all features into a single array
        features = np.array([[bmi, physical_health, mental_health, sleep_time,
                              smoking, stroke, diff_walking, sex, age_category, gen_health]])
 # Predict using the loaded model
        probability = model.predict_proba(features)[0]  # Get the probability for both classes
        percentage_probability = "{:.2f}%".format(probability[1] * 100)

        # Get the prediction (0 for no disease, 1 for disease)
        prediction = model.predict(features)[0]
        prediction_text = 'Healthy' if prediction == 0 else 'At Risk'

        # Determine the risk message
        risk_message = 'Low risk of heart disease.' if prediction == 0 else 'High risk of heart disease.'

    # Pass additional information to the template
    return render_template('index.html', prediction=prediction_text,
                           percentage_probability=percentage_probability,
                           risk_message=risk_message)

if __name__ == '__main__':
    app.run(debug=True)