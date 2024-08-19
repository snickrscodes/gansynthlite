from model import Model, Generator, Discriminator
import tensorflow as tf
import keras
import random
import numpy as np
import nnops as ops
# on my machine was only able to complete around 6000/50000 training steps
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

agent = Model()
agent.train()
agent.generate_examples(16)
