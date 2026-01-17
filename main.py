from flask import Flask, render_template, request
import tensorflow as tf
import os

from utils import load_image, decode_predictions

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(
    "model/synth90k_crnn.keras",
    compile=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            img = load_image(image_path)
            preds = model(img, training=False)
            result = decode_predictions(preds)[0]

    return render_template("index.html", result=result, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
