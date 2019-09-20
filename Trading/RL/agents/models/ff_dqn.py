import tensorflow as tf
import numpy as np

from agents.models.common import fc, conv

settings = {
            #Env:
            "gamma" : 0.99,
            #Training params
            "minibatch_size" : 256,
            "lr" : lambda x : (1-x) * 1.0 * 10**-5,
            "nn_type" : "conv", #"dense",
            }

class ff_model:
    def __init__(self, env, session, scope):
        self.session = session
        self.env = env
        with tf.variable_scope(scope):
            dummy_state = self.state_unload(env.get_state())
            if settings["nn_type"] == "conv":
                self.input_seq_tf = tf.placeholder(tf.float16, shape=[None, *dummy_state[0].shape[1:]])
                self.input_vec_tf = tf.placeholder(tf.float16, shape=[None, *dummy_state[1].shape[1:]])
            else:
                self.input_vec_tf = tf.placeholder(tf.float16, shape=[None, *dummy_state[0].shape])

            self.target_q_tf = tf.placeholder(tf.float16, shape=[None,])
            self.action_tf = tf.placeholder(tf.int32, shape=[None,])

            if settings["nn_type"] == "dense":
                self.output_tf = fc(
                                    self.input_vec_tf,
                                    n_hidden=4,
                                    hidden_size=4096,
                                    out_size=len(env.actions),
                                    )
            if settings["nn_type"] == "conv":
                self.output_tf = conv(
                                      self.input_seq_tf,
                                      self.input_vec_tf,
                                      n_hidden=4,
                                      n_filters=32,
                                      filter_size=32,
                                      out_size=len(env.actions),
                                    )
            self.lr_tf = tf.placeholder(tf.float16, shape=[], name="lr")
            self.training_ops, self.loss_tf = self.create_training_and_loss_ops()
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.assign_ops, assign_values = self.create_weight_setting_ops()
            self.training_outputs = [self.loss_tf, self.lr_tf]

    @property
    def vars(self):
        return self.all_variables

    def eval(self, s, internal_call=False):
        s = s if internal_call else self.state_unload(s)
        if settings["nn_type"] == "conv":
            s_seq, s_vec = s
        else:
            s_vec = s
        feed_dict = {}
        feed_dict[self.input_vec_tf] = s_vec
        if settings["nn_type"] == "conv":
            feed_dict[self.input_seq_tf] = s_seq
        q = self.session.run(self.output_tf, feed_dict=feed_dict)
        return q

    def train(self, batch, ref=None, progress=0.0, printer=None):
        if ref is None: ref = self
        #Get batch into np-arrays
        n = len(batch)
        n_batches = n/settings["minibatch_size"]
        _s,_a,_r,_sp,_d = list(zip(*batch))
        if settings["nn_type"] == "dense":
            s_vec  = np.concatenate( list(map(self.state_unload, _s )),  axis=0)
            sp_vec = np.concatenate( list(map(self.state_unload, _sp)),  axis=0)
        if settings["nn_type"] == "conv":
            tmp = list(zip(*list(map(self.state_unload, _s ))))
            s_seq,  s_vec  = np.concatenate(tmp[0], axis=0), np.concatenate(tmp[1], axis=0)
            tmp = list(zip(*list(map(self.state_unload, _sp))))
            sp_seq, sp_vec = np.concatenate(tmp[0], axis=0), np.concatenate(tmp[1], axis=0)
        a  = np.concatenate( _a                               ,  axis=0)
        r  = np.concatenate( _r                               ,  axis=0)
        d  = np.concatenate( _d                               ,  axis=0)
        sp = (sp_seq, sp_vec) if settings["nn_type"] == "conv" else s_vec
        ref_q = ref.eval( sp, internal_call=True)
        ref_v = np.max(ref_q, axis=-1)
        target = r + settings["gamma"] * ref_v * (1-d)

        #Training
        rets = []
        for idx in range(0,n, settings["minibatch_size"]):
            high = min(len(batch), idx+settings["minibatch_size"])
            s = (s_seq[idx:high,:], s_vec[idx:high,:]) if settings["nn_type"] == "conv" else s_vec[idx:high,:]
            ret = self.update(
                               s,
                               a[idx:high],
                               target[idx:high],
                               lr=settings["lr"](progress)
                               )
            rets.append([r/n_batches for r in ret])
            if printer is not None: printer.tick()

        tot_ret = list( map( sum, zip(*rets) ) )
        return dict(zip(self.training_outputs,tot_ret))

    def update(self, s, a, target, lr=None):
        assert lr is not None, "specify learning rate for training!"
        assert type(s) is np.array or type(target is np.array), "only train on ndarrays"
        feed_dict = {
                     self.action_tf : a,
                     self.target_q_tf : target,
                     self.lr_tf : lr,
                    }
        if settings["nn_type"] == "conv":
            feed_dict[self.input_seq_tf] = s[0]
            feed_dict[self.input_vec_tf] = s[1]
        else:
            feed_dict[self.input_vec_tf] = s
        ret, _ = self.session.run([self.training_outputs, self.training_ops], feed_dict=feed_dict)
        return ret

    def create_training_and_loss_ops(self):
        mask_tf = tf.cast(tf.one_hot(self.action_tf, 3), dtype=tf.float16)
        masked_output_tf = tf.multiply( self.output_tf, mask_tf)
        diff_tf = tf.reduce_sum(masked_output_tf, axis=-1) - self.target_q_tf
        loss_tf = tf.reduce_mean( tf.abs(diff_tf), name="loss" )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_tf)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_tf)
        training_ops = optimizer.minimize(loss_tf)
        return training_ops, loss_tf

    def get_weights(self):
        return self.session.run(self.all_variables)
    def set_weights(self, weights):
        feed_dict = dict(zip(self.assign_ops, weights))
        self.session.run(self.assign_ops, feed_dict=feed_dict)

    def create_weight_setting_ops(self):
        assign_ops = []
        assign_values = []
        for var in self.all_variables:
            shape, dtype = var.shape, var.dtype
            assign_val_placeholder_tf = tf.placeholder(shape=shape, dtype=dtype)
            assign_op_tf = var.assign(assign_val_placeholder_tf)
            assign_ops.append(assign_op_tf)
            assign_values.append(assign_val_placeholder_tf)
        return assign_ops, assign_values

    def state_unload(self, s):
        if type(s) is np.array or type(s) is np.ndarray:
            assert False, "I think you want to be aware if this happens, so I terminate here..."
            return s
        if type(s) is list:
            return [self.state_unload(x) for x in s]
        if settings["nn_type"] == "dense":
            return np.concatenate(
                                    (
                                     s["customer"].reshape((1,-1)),
                                     s["market"].to_numpy().reshape((1,-1))
                                    ),
                                    axis=-1
                                 )
        if settings["nn_type"] == "conv":
            seq = s["market"].to_numpy().reshape((1,self.env.state_len,6))
            vec = s["customer"].reshape((1, -1))
            return seq, vec
