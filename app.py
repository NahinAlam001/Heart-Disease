import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        int_features = [float(data['biking']), float(data['smoking'])]
        features = [np.array(int_features)]
        prediction = model.predict(features)

        output = round(prediction[0], 2)
        return jsonify({'prediction_text': f'Percent Population with Heart Disease: {output}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
