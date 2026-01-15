import tensorflow as tf
import numpy as np
import string

IMG_HEIGHT = 32
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


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    new_w = tf.cast(w * IMG_HEIGHT / h, tf.int32)
    img = tf.image.resize(img, (IMG_HEIGHT, new_w))
    img = tf.cast(img, tf.float32) / 255.0

    return tf.expand_dims(img, axis=0)


def decode_predictions(pred):
    input_len = tf.fill([pred.shape[0]], pred.shape[1])

    decoded, _ = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )

    texts = []
    for seq in decoded[0]:
        seq = tf.boolean_mask(seq, seq != -1)
        text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode()
        texts.append(text)

    return texts
