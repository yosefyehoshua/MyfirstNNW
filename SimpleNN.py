from numpy import exp, array, random, dot
import tensorflow


# todo:
# 1. sigmoid
# 2. derivative of sigmoid for GD
# 3. iterating for training


class MyNN():

    def __init__(self):

        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1))


    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def derivativeOfSigmoid(self, func):
        return func * (1 - func)



    def think(self, input, weights):
        return self.sigmoid()