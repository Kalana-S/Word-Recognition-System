from flask import Flask, render_template, request
import tensorflow as tf
import os

from utils import confidence_based_predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

baseline_model = tf.keras.models.load_model("model/baseline_crnn.keras")
transfer_model = tf.keras.models.load_model("model/transfer_learning_crnn.keras")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    model_used = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            output = confidence_based_predict(
                image_path,
                baseline_model,
                transfer_model
            )

            result = output["text"]
            confidence = output["confidence"]
            model_used = output["model"]

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        model_used=model_used,
        image=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)
