import numpy as np
import tensorflow as tf
from scipy.stats.mstats import gmean

from agents.models.ff_dqn import ff_model
from agents.models.common import printer
from agents.replaybuffers import *

settings = {
            #Agent:
            "epsilon" : lambda t_now, t_max : 0.1*np.sqrt((t_max+1)/(1 * t_now + 1)),""
            "print_every_n_steps" : 100,
            "train_after_n_steps" : 64,
            "n_samples_to_train_on" : 2048,
            "save_every_n_steps" : 100000,
            "reference_update_after_n_updates" : 5,
            "reference_update_mode" : "propagate", #swap or propagate
            "replay_buffer_type" : experience_replay_buffer,
            "saver_dir" : "agents/models/saved/",
            "model_type" : ff_model,
            }

class agent:
    def __init__(self, name, session, env=None, eval_env=None):
        assert env is not None, "agent > you need to specify: agent(env=...)"
        self.env = env
        self.eval_env = env if eval_env is None else eval_env
        self.name = name
        self.session = session
        self.experiences = settings["replay_buffer_type"](max_size=10000)
        self.model_0 = settings["model_type"](env, session, "model0")
        self.model_1 = settings["model_type"](env, session, "model1")
        self.saver = tf.train.Saver(self.model_0.vars + self.model_1.vars)
        self.printer = printer()
        ##
        ## Safety guard!
        if type(self.experiences) is prioritized_experience_replay:
            assert False, "prioritized experience replay not implemented yet"

    def eval(self, t):
        s = self.env._reset(t)
        q = self.get_q(s)
        return q

    def evaluate_model(self, n_steps, verbose=True):
        #Trajectory variables...
        first, d, R, n_episodes = True, True, 0, 0
        stats = {
                    "market_inc" : [],
                    "value_inc"  : [],
                    "relative_value_inc" :[],
                    "reward" : [],
                    "action_count" : [1,1,1], #approximately 0
                    "action_entropy" : 0,
                    "q_spread" : [],
                    "q_max" : [],
                    "q_min" : [],
                }
        #Main loop
        n_filtered = 0
        moving_avg_spread = 0.1
        def stats_summary(s):
            summary = {}
            for x in s:
                if x in ["action_entropy", "action_count"]:
                    _p = np.array(s["action_count"])
                    p = _p/_p.sum()
                    summary["action_entropy"] = ( -p * np.log(p) ).sum()
                else:
                    summary[x] = np.mean(np.array(s[x]).ravel())
            return summary

        for t in range(n_steps):
            #Reset?
            if d:
                if not first:
                    n_episodes += 1
                    market_inc = self.env.market.normalized_price - start_price
                    value_inc = self.env.customer.normalized_value - start_val
                    stats["market_inc"].append(market_inc)
                    stats["value_inc"].append(value_inc)
                    stats["reward"].append(R)
                    stats["relative_value_inc"].append(value_inc - market_inc)
                s_prime = self.env.reset()
                start_price = float(self.env.market.normalized_price)
                start_val = float(self.env.customer.normalized_value)
                R = 0.0
            #Perform action!
            s = s_prime
            a, q = self.get_action(s, q=True)
            ###
            ###
            delta = np.amax(q, axis=-1) - np.amin(q, axis=-1)
            stats["action_count"][a[0]] += 1
            stats["q_spread"].append(delta.ravel())
            stats["q_max"].append(np.amax(q, axis=-1).ravel())
            stats["q_min"].append(np.amin(q, axis=-1).ravel())
            ###
            r, s_prime, d, info = self.env.perform_action(a, eval=True)
            #Store experience
            R += r
            first = False
            if t%100 == 0:
                print("stats")
                for kw,val in stats_summary(stats).items():
                    print("\t", kw, "\t", val)
        summary = stats_summary(stats)
        if verbose:
            print("{} ::::: Model performance:".format(self.name))
            for x in summary:
                print(np.mean(summary[x]))
        return summary

    def train(self, n_steps):
        #Update timers...
        time_since_ref_update, time_since_training, tot_reward = 0, 0, 0.0

        #Trajectory variables...
        d, R = True, 0.0

        #Main loop
        for t in range(n_steps):

            #Reset?
            if d:
                s_prime = self.env.reset()
                tot_reward += R
                print("Episode done! \n TOTAL REWARD = ", R, "\n", tot_reward/(t+1) , "<-- avg ep reward")
                R = 0.0

            #Perform action!
            s = s_prime
            a = self.get_action(s, epsilon=settings["epsilon"](t, n_steps))
            r, s_prime, d, info = self.env.perform_action(a)

            #Store experience
            R += r
            experience = (s,a,r,s_prime,d)
            self.experiences.add(experience)

            #Train?
            time_since_training += 1
            if (t+1)%settings["save_every_n_steps"] == 0:
                self.save_model(global_step=t+1)
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

    def get_action(self,state, epsilon=None, q=False):
        n = 1 if type(state) is dict else len(state)
        qs = self.get_q(state)
        a = np.argmax(qs, axis=-1).reshape((n,))
        if epsilon is not None:
            if np.random.uniform() < epsilon:
                a = np.random.randint(self.env.n_actions, size=(n,))
        return (a, qs) if q else a

    def get_q(self,state):
        qs = self.model_0.eval(state)
        return qs

    def perform_reference_update(self):
        if settings["reference_update_mode"] == "swap":
            _tmp = self.model_0
            self.model_0 = self.model_1
            self.model_1 = _tmp
        if settings["reference_update_mode"] == "propagate":
            self.model_1.set_weights( self.model_0.get_weights() )

    def save_model(self, global_step=0):
        self.saver.save(self.session, settings["saver_dir"]+self.name, global_step=global_step)
    def load_model(self, version):
        self.saver.restore(self.session, settings["saver_dir"]+self.name+"-"+version)

    @property
    def current_eval(self):
        return  dict(zip( [self.env.actions[i] for i in range(self.env.n_actions)], self.eval(self.env.t)[0] ))
