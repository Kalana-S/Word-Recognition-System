from flask import Flask, render_template, request
import tensorflow as tf
import os

from utils import load_image, decode_predictions

app = Flask(__name__)

model = tf.keras.models.load_model(
    "model/synth90k_crnn.keras"
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join("static", file.filename)
        file.save(path)

        img = load_image(path)
        preds = model(img, training=False)
        result = decode_predictions(preds)[0]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
