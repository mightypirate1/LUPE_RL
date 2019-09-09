import numpy as np
import tensorflow as tf

from agents.common import fc
from agents.replaybuffers import *

settings = {
            #Env:
            "gamma" : 0.99,
            #Agent:
            "epsilon" : lambda t_now, t_max : 0.01*np.sqrt((t_max+1)/(0.3 * t_now + 1)),
            "print_every_n_steps" : 1,
            "train_after_n_steps" : 16,
            "n_samples_to_train_on" : 16,
            "reference_update_after_n_updates" : 1,
            "minibatch_size" : 64,
            "lr" : 1.0 * 10**-4,
            "reference_update_mode" : "propagate", #swap or propagate
            "replay_buffer_type" : experience_replay_buffer,
            }

class model:
    def __init__(self, env, session, scope):
        self.sess = session
        with tf.variable_scope(scope):
            self.shape = self.state_unload(env.get_state()).shape[1:]
            self.input_tf = tf.placeholder(tf.float16, shape=[None, *self.shape])
            self.target_q_tf = tf.placeholder(tf.float16, shape=[None,])
            self.action_tf = tf.placeholder(tf.int32, shape=[None,])
            self.output_tf = fc(self.input_tf)
            self.training_ops, self.loss_tf = self.create_training_and_loss_ops()
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.assign_ops, assign_values = self.create_weight_setting_ops()

    def eval(self, s):
        feed_dict = {
                     self.input_tf : self.state_unload(s),
                     }
        q = self.sess.run(self.output_tf, feed_dict=feed_dict)
        return q

    def update(self, s, a, target):
        assert type(s) is np.array or type(target is np.array), "only train on ndarrays"
        feed_dict = {
                     self.input_tf : s,
                     self.action_tf : a,
                     self.target_q_tf : target,
                    }
        loss, _ = self.sess.run([self.loss_tf, self.training_ops], feed_dict=feed_dict)
        return None, loss

    def create_training_and_loss_ops(self):
        mask_tf = tf.cast(tf.one_hot(self.action_tf, 3), dtype=tf.float16)
        masked_output_tf = tf.multiply( self.output_tf, mask_tf)
        diff_tf = tf.reduce_sum(masked_output_tf, axis=-1) - self.target_q_tf
        loss_tf = tf.reduce_mean( tf.abs(diff_tf) )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=settings["lr"])
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

class agent:
    def __init__(self, session, env=None):
        assert env is not None, "agent > you need to specify: agent(env=...)"
        self.env = env
        self.sess = session
        self.experiences = settings["replay_buffer_type"](max_size=10000)
        self.model_0 = model(env, session, "model0")
        self.model_1 = model(env, session, "model1")

    def train(self, n_steps):
        #Update timers...
        time_since_ref_update, time_since_training = 0, 0
        #Trajectory variables...
        d, R = True, 0

        #Main loop
        for t in range(n_steps):
            #Reset?
            if d:
                print("Episode done! \n TOTAL REWARD = ", R, "\n")
                s_prime = self.env.reset()
            #Perform action!
            s = s_prime
            a = self.get_action(s, epsilon=settings["epsilon"](t, n_steps))
            r, s_prime, d = self.env.perform_action(a)
            #Store experience
            R += r
            self.experiences.add((s,a,r,s_prime))
            #Train?
            time_since_training += 1
            if time_since_training > settings["train_after_n_steps"] and len(self.experiences) > settings["n_samples_to_train_on"]:
                time_since_training = 0
                ref_update = (time_since_ref_update > settings["reference_update_after_n_updates"])
                self.update_model(update_reference=ref_update, prints=(time_since_ref_update, settings["reference_update_after_n_updates"]))
                time_since_ref_update +=1
                if ref_update: time_since_ref_update = 0

            #Prints...
            if t % settings["print_every_n_steps"] == 0:
                print("step ", t, " : ", "|| a =", a,  "||| r =",r , "||      ", s["customer"], " epsilon=",settings["epsilon"](t, n_steps))

    def update_model(self, batch_size=settings["n_samples_to_train_on"], update_reference=False, prints=None):
        if update_reference:
            self.perform_reference_update()
        ref_string = "" if prints is None else "<"+"="*prints[0]+"-"*(prints[1]-prints[0])+">"
        tot_loss = 0.0
        batch = self.experiences.get_sample(batch_size)
        new_prios = np.zeros((batch_size,1))
        if type(self.experiences) is prioritized_experience_replay:
            batch, weights, filter = batch
        _s  = np.concatenate([self.model_0.state_unload(x[0]) for x in batch ], axis=0)
        _a =  np.concatenate([  x[1]                          for x in batch ], axis=0)
        _r =  np.concatenate([ [x[2]]                         for x in batch ], axis=0)
        _sp = np.concatenate([self.model_0.state_unload(x[3]) for x in batch ], axis=0)
        ref_q = self.model_1.eval(_sp)
        ref_v = np.max(ref_q, axis=-1)
        target = _r + settings["gamma"] * ref_v
        print("[", end='', flush=True)
        for idx in range(0,batch_size, settings["minibatch_size"]):
            high = min(len(batch), idx+settings["minibatch_size"])
            # print(_s[idx:high,:].shape, _a[idx:high].shape, target[idx:high].shape)
            _new_prios, loss = self.model_0.update(_s[idx:high,:], _a[idx:high], target[idx:high])
            new_prios[idx:high] = _new_prios
            tot_loss += loss
            print("|",end="", flush=True)
        if type(self.experiences) is prioritized_experience_replay:
            blah
        print("] {} | {}".format(ref_string, tot_loss), flush=True)

    def get_action(self,state, epsilon=None):
        if type(state) is dict:
            n = 1
        else:
            n = len(state)
        if epsilon is not None:
            if np.random.uniform() < epsilon:
                return np.random.randint(3, size=(n,))
        qs = self.model_0.eval(state)
        return np.argmax(qs, axis=-1).reshape((n,))

    def perform_reference_update(self):
        if settings["reference_update_mode"] == "swap":
            _tmp = self.model_0
            self.model_0 = self.model_1
            self.model_1 = _tmp
        if settings["reference_update_mode"] == "propagate":
            self.model_1.set_weights( self.model_0.get_weights() )
