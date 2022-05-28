from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('data/spam_classifier.pkl', 'rb'))
cv = pickle.load(open('data/cv-transform.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    messages = req['messages']
    response = {'messages': []}
    for sms in messages:
        data = [sms['message']]
        vector = cv.transform(data).toarray()
        prediction = model.predict(vector)
        result = {'sender': sms['sender'], 'message': sms['message'], 'is_spam': bool(prediction[0])}
        response['messages'].append(result)

    return jsonify(response)


if __name__ == '__main__':
    app.run()
