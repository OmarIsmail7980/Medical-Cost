from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('medical.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    bmi = request.form['bmi']
    children = request.form['children']
    smoker = request.form['smoker']
    region = request.form['region']
     
    arr = np.array([[age,sex,bmi,children,smoker,region]])
    
    prediction = model.predict(arr)
    
    prediction = np.round(prediction[0], 2)
    
    return render_template("index.html", prediction_text=
                           'Your medical cost is ${}'.format(prediction))
    
    
if __name__ == "__main__":
    app.run(debug=True)

