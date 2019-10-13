import tensorflow as tf

import env.builder
from agents.dqn import agent

project = "ape"
env_str = "SingleStock-v0"

###
load  = None
train = True
eval  = True
demo  = False

###
# load  = "700000"
# train = False

e      = env.builder.build(env_str)
# e_eval = env.builder.build(env_str, eval=True)

agent_str = env_str + "/" + project
with tf.Session() as session:
    a = agent( agent_str, session, env=e)
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
