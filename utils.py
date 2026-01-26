import tensorflow as tf
import numpy as np

CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=CHARS,
    mask_token=None,
    oov_token=None
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary()[1:],
    invert=True
)

IMG_HEIGHT = 32
IMG_WIDTH_NEW = 256

def preprocess_old(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_w = tf.cast(w * IMG_HEIGHT / h, tf.int32)

    img = tf.image.resize(img, (IMG_HEIGHT, new_w))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)


def preprocess_new(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH_NEW))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)

def ctc_decode_with_confidence(pred):
    """
    pred: (1, T, C)
    """
    time_steps = pred.shape[1]
    input_len = tf.fill([1], time_steps)

    decoded, _ = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )

    seq = decoded[0][0]
    seq = tf.boolean_mask(seq, seq != -1)

    text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode()

    probs = tf.reduce_max(pred, axis=-1)  
    confidence = tf.reduce_mean(probs).numpy()

    return text, confidence

CONF_THRESHOLD = 0.85

def hybrid_predict(image_path, old_model, new_model):
    """
    1. Try old model first
    2. If confidence is low, fallback to new model
    """

    img_old = preprocess_old(image_path)
    pred_old = old_model(img_old, training=False)
    text_old, conf_old = ctc_decode_with_confidence(pred_old)

    if conf_old >= CONF_THRESHOLD:
        return {
            "text": text_old,
            "model": "old",
            "confidence": float(conf_old)
        }

    img_new = preprocess_new(image_path)
    pred_new = new_model(img_new, training=False)
    text_new, conf_new = ctc_decode_with_confidence(pred_new)

    return {
        "text": text_new,
        "model": "new",
        "confidence": float(conf_new)
    }
