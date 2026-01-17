import tensorflow as tf

IMG_HEIGHT = 32
IMG_WIDTH = 256

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
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)

def decode_predictions(pred):
    batch_size = tf.shape(pred)[0]
    time_steps = tf.shape(pred)[1]

    input_len = tf.fill([batch_size], time_steps)

    decoded, _ = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )

    texts = []
    for seq in decoded[0]:
        seq = tf.boolean_mask(seq, seq != -1)
        text = tf.strings.reduce_join(num_to_char(seq)).numpy().decode("utf-8")
        texts.append(text)

    return texts
