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

def preprocess_baseline(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_w = tf.cast(w * IMG_HEIGHT / h, tf.int32)

    img = tf.image.resize(img, (IMG_HEIGHT, new_w))
    img = tf.cast(img, tf.float32) / 255.0

    return tf.expand_dims(img, axis=0)

def preprocess_transfer(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH_NEW))
    img = tf.cast(img, tf.float32) / 255.0

    return tf.expand_dims(img, axis=0)

def decode_with_confidence(pred):
    batch_size = tf.shape(pred)[0]
    time_steps = tf.shape(pred)[1]

    input_len = tf.fill([batch_size], time_steps)

    decoded, log_probs = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )

    seq = decoded[0][0]
    log_prob = log_probs[0][0]
    confidence = float(tf.exp(-log_prob).numpy())
    seq = tf.boolean_mask(seq, seq != -1)
    text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode("utf-8")

    return text, round(confidence, 3)

def confidence_based_predict(image_path, baseline_model, transfer_model):
    img_base = preprocess_baseline(image_path)
    pred_base = baseline_model(img_base, training=False)
    text_base, conf_base = decode_with_confidence(pred_base)
    img_tl = preprocess_transfer(image_path)
    pred_tl = transfer_model(img_tl, training=False)
    text_tl, conf_tl = decode_with_confidence(pred_tl)

    if conf_tl > conf_base:
        return {
            "text": text_tl,
            "confidence": conf_tl,
            "model": "transfer_learning"
        }
    else:
        return {
            "text": text_base,
            "confidence": conf_base,
            "model": "baseline"
        }
