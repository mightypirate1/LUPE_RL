import tensorflow as tf
import numpy as np

from agents.models.common import fc

settings = {
            #Env:
            "gamma" : 0.99,
            #Training params
            "minibatch_size" : 64,
            "lr" : lambda x : (1-x) * 1.0 * 10**-4,
            }

class ff_model:
    def __init__(self, env, session, scope):
        self.sess = session
        with tf.variable_scope(scope):
            self.shape = self.state_unload(env.get_state()).shape[1:]
            self.input_tf = tf.placeholder(tf.float16, shape=[None, *self.shape])
            self.target_q_tf = tf.placeholder(tf.float16, shape=[None,])
            self.action_tf = tf.placeholder(tf.int32, shape=[None,])
            self.output_tf = fc(self.input_tf)
            self.lr_tf = tf.placeholder(tf.float16, shape=[])
            self.training_ops, self.loss_tf = self.create_training_and_loss_ops()
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.assign_ops, assign_values = self.create_weight_setting_ops()
            self.training_outputs = [self.loss_tf,]

    def eval(self, s, internal_call=False):
        feed_dict = {
                     self.input_tf : s if internal_call else self.state_unload(s),
                     }
        q = self.sess.run(self.output_tf, feed_dict=feed_dict)
        return q

    def train(self, batch, ref=None, progress=0.0, printer=None):
        if ref is None: ref = self
        #Get batch into np-arrays
        n = len(batch)
        _s,_a,_r,_sp,_d = list(zip(*batch))
        s  = np.concatenate( list(map(self.state_unload, _s )),  axis=0)
        a  = np.concatenate( _a                               ,  axis=0)
        r  = np.concatenate( _r                               ,  axis=0)
        sp = np.concatenate( list(map(self.state_unload, _sp)),  axis=0)
        d  = np.concatenate( _d                               ,  axis=0)
        ref_q = ref.eval(sp, internal_call=True)
        ref_v = np.max(ref_q, axis=-1)
        ''' TODO: Think carefully about the next line... '''
        target = r + settings["gamma"] * ref_v # * (1-d)

        #Training
        rets = []
        for idx in range(0,n, settings["minibatch_size"]):
            high = min(len(batch), idx+settings["minibatch_size"])
            ret = self.update(s[idx:high,:], a[idx:high], target[idx:high], lr=settings["lr"](progress))
            rets.append(ret)
            if printer is not None: printer.tick()

        tot_ret = list( map( sum, zip(*rets) ) )
        return dict(zip(self.training_outputs,tot_ret))

    def update(self, s, a, target, lr=None):
        assert lr is not None, "specify learning rate for training!"
        assert type(s) is np.array or type(target is np.array), "only train on ndarrays"
        feed_dict = {
                     self.input_tf : s,
                     self.action_tf : a,
                     self.target_q_tf : target,
                     self.lr_tf : lr,
                    }
        ret, _ = self.sess.run([self.training_outputs, self.training_ops], feed_dict=feed_dict)
        return ret

    def create_training_and_loss_ops(self):
        mask_tf = tf.cast(tf.one_hot(self.action_tf, 3), dtype=tf.float16)
        masked_output_tf = tf.multiply( self.output_tf, mask_tf)
        diff_tf = tf.reduce_sum(masked_output_tf, axis=-1) - self.target_q_tf
        loss_tf = tf.reduce_mean( tf.abs(diff_tf), name="loss" )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_tf)
        # optimizer = tf.train.AdamOptimizer(learning_rate=settings["lr"])
        training_ops = optimizer.minimize(loss_tf)
        return training_ops, loss_tf

    def get_weights(self):
        return self.sess.run(self.all_variables)
    def set_weights(self, weights):
        feed_dict = dict(zip(self.assign_ops, weights))
        self.sess.run(self.assign_ops, feed_dict=feed_dict)

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
        assert type(s) is not tuple, "!!!!!!"
        if type(s) is np.array or type(s) is np.ndarray:
            assert False, "I think you want to be aware if this happens, so I terminate here..."
            return s
        if type(s) is list:
            return [self.state_unload(x) for x in s]
        return np.concatenate(
                                (
                                 s["customer"].reshape((1,-1)),
                                 s["market"].to_numpy().reshape((1,-1))
                                ),
                                axis=-1
                             )
