import tensorflow as tf

from env import env
from agents.dqn import agent

###
load  = None
train = True
eval  = True
demo  = False

###
# load  = "ape-900000"
# train = False

e = env()
# e = env(file="LUPE.ST.csv")

with tf.Session() as session:
    a = agent( "ape", session, env=e)
    session.run(tf.global_variables_initializer())
    if load is not None:
        a.load_model(load)
    if train:
        a.train(1000000)
    if eval:
        stats = a.evaluate_model(10000)
    if demo:
        a.env = env(interactive=True)
        print("Current evaluation is:")
        print(a.current_eval)
        a.env.reset()
