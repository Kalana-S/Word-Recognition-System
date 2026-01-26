from flask import Flask, render_template, request
import tensorflow as tf
import os

from utils import hybrid_predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

old_model = tf.keras.models.load_model("model/baseline_crnn.keras")
new_model = tf.keras.models.load_model("model/transfer_learning_crnn.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        output = hybrid_predict(image_path, old_model, new_model)
        result = output["text"]
        confidence = output["confidence"]

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
