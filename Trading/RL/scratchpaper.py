import tensorflow as tf

from env import env
from agents.dqn import agent

e = env()
with tf.Session() as session:
    a = agent(session, env=e)
    session.run(tf.global_variables_initializer())
    a.train(10000)
