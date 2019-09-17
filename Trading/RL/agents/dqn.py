import numpy as np
import tensorflow as tf

from agents.models.ff import ff_model
from agents.models.common import printer
from agents.replaybuffers import *

settings = {
            #Agent:
            "epsilon" : lambda t_now, t_max : 0.01*np.sqrt((t_max+1)/(0.3 * t_now + 1)),
            "print_every_n_steps" : 100,
            "train_after_n_steps" : 64,
            "n_samples_to_train_on" : 1024,
            "reference_update_after_n_updates" : 5,
            "reference_update_mode" : "propagate", #swap or propagate
            "model_type" : ff_model,
            "replay_buffer_type" : experience_replay_buffer,
            }

class agent:
    def __init__(self, session, env=None):
        assert env is not None, "agent > you need to specify: agent(env=...)"
        self.env = env
        self.sess = session
        self.experiences = settings["replay_buffer_type"](max_size=10000)
        self.model_0 = settings["model_type"](env, session, "model0")
        self.model_1 = settings["model_type"](env, session, "model1")
        self.printer = printer()
        ##
        ## Safety guard!
        if type(self.experiences) is prioritized_experience_replay:
            assert False, "prioritized experience replay not implemented yet"

    def eval(self, t):
        s = self.env._reset(t)
        q = self.get_q(s)
        return q

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
            self.experiences.add((s,a,r,s_prime,d))
            #Train?
            time_since_training += 1
            if time_since_training > settings["train_after_n_steps"] and len(self.experiences) > settings["n_samples_to_train_on"]:
                time_since_ref_update += 1
                if (time_since_ref_update > settings["reference_update_after_n_updates"]):
                    self.perform_reference_update()
                    time_since_ref_update = 0
                self.update_model(steps=(t,n_steps), ref_updates=(time_since_ref_update, settings["reference_update_after_n_updates"]))
                time_since_training = 0

            #Prints...
            if t % settings["print_every_n_steps"] == 0:
                print("step ", t, " : ", "|| a =", a,  "||| r =",r , "||      ", s["customer"], " epsilon=",settings["epsilon"](t, n_steps))

    def update_model(self, steps=(0,1), batch_size=settings["n_samples_to_train_on"], ref_updates=None):
        batch = self.experiences.get_sample(batch_size)
        if type(self.experiences) is prioritized_experience_replay:
            batch, weights, filter = batch
        self.printer.batch_start()
        ret = self.model_0.train(batch, progress=steps[0]/steps[1], ref=self.model_1, printer=self.printer)
        self.printer.batch_end(ref_updates=ref_updates, returns=ret)

    def get_action(self,state, epsilon=None):
        if type(state) is dict:
            n = 1
        else:
            n = len(state)
        qs = self.get_q(state)
        return np.argmax(qs, axis=-1).reshape((n,))

    def get_q(self,state, epsilon=None):
        if type(state) is dict:
            n = 1
        else:
            n = len(state)
        if epsilon is not None:
            if np.random.uniform() < epsilon:
                return np.random.randint(3, size=(n,))
        qs = self.model_0.eval(state)
        return qs

    def perform_reference_update(self):
        if settings["reference_update_mode"] == "swap":
            _tmp = self.model_0
            self.model_0 = self.model_1
            self.model_1 = _tmp
        if settings["reference_update_mode"] == "propagate":
            self.model_1.set_weights( self.model_0.get_weights() )

    @property
    def current_eval(self):
        return  dict(zip( self.env.actions, self.eval(self.env.t)[0] ))
