import tensorflow as tf

def fc(x, n_hidden=2, hidden_size=2048, out_size=3):
    x = x-1
    for _ in range(n_hidden):
        x = tf.layers.dense(x, 2048, activation=tf.nn.elu)
    ret = tf.layers.dense(x, out_size, activation=tf.keras.activations.linear)
    return ret

#has 2 heads... one for conv, one for vec
def conv(in_seq, in_vec, n_hidden=2, n_filters=32, filter_size=8, out_size=3):
    x = in_seq-1
    for _ in range(n_hidden):
        x = tf.layers.conv1d(
                             x,
                             n_filters,
                             filter_size,
                             padding='same',
                             activation=tf.nn.elu,
                             )
        x = tf.layers.max_pooling1d(x, 2, 2)
    x = tf.contrib.layers.flatten(x)
    x = tf.concat([x, in_vec], -1)
    ret = fc(x , hidden_size=512, n_hidden=1, out_size=out_size)
    return ret



class printer:
    def __init__(self):
        pass
    def tick(self):
        print("|",end="", flush=True)
    def batch_start(self):
        print("[", end='', flush=True)
    def batch_end(self, steps=None, ref_updates=None, returns=None):
        ref_string = "" if ref_updates is None else "<"+"="*ref_updates[0]+"-"*(ref_updates[1]-ref_updates[0])+">"
        #returns is expected to be a dictionary mapping tensors to floats, eg. loss_tf : 0.123
        returns_string = ""
        if returns is not None:
            for tensor in returns:
                returns_string += "("+tensor.name+" "+str(returns[tensor])+") "
        print("] {} | {}".format(ref_string, returns_string), flush=True)
