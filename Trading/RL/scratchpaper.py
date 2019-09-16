import tensorflow as tf

from env import env
from agents.dqn import agent

load = False
train = True
eval = False
demo = False

e = env()
# e = env(file="LUPE.ST.csv")

with tf.Session() as session:
    a = agent(session, env=e)
    session.run(tf.global_variables_initializer())
    if load:
        exit("not implemented")
    if train:
        a.train(1000000)
    if eval:
        a.env = env(interactive=True)
        print("Current evaluation is:")
        print(a.current_eval)
    if demo:
        a.env.reset()
