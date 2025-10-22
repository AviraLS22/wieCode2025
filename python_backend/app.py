from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model, vectorizer = pickle.load(open("model/intent_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")

    X = vectorizer.transform([query])
    intent = model.predict(X)[0]

    return jsonify({"intent": intent})

if __name__ == "__main__":
    app.run(debug=True)
