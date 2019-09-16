import tensorflow as tf

def fc(x, n_hidden=2, out_size=3):
    x = x-1
    for _ in range(n_hidden):
        x = tf.layers.dense(x, 2048, activation=tf.nn.elu)
    ret = tf.layers.dense(x, out_size, activation=tf.keras.activations.linear)
    return ret
