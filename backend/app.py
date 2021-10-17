from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("forest.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)

    if prediction[0] == 1:
        return render_template('forest.html', pred='You are at risk for heart attack. Please visit your cardiologist')
    else:
        return render_template('forest.html', pred='You are currently not at risk for heart attack.')


if __name__ == '__main__':
    app.run(debug=True)
