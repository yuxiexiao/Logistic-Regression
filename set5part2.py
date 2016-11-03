n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                import numpy as np
import random
import math
from numpy import linalg as LA

class LogRegress:
    ''' Class for Logistic Regression
    '''

    def __init__(self, n):
        ''' Generates a random target function f, and generates
            n random data points x from [-1, 1] x [-1, 1].
        '''
        x1, x2, y1, y2 = [np.random.uniform(-1.0, 1.0) for i in range(4)]
        self.target = np.array([x2*y1 - x1*y2, y2-y1, x1-x2])
        self.data = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0)) for i in range(n)])
        self.y = np.array([int(np.sign(self.target.transpose().dot(x)))
                      for x in self.data])

        self.w = np.array([0.0, 0.0, 0.0])
        self.prev = np.array([0.0, 0.0, 0.0])
        self.epoch = 0


    def getGrad(self, n):
        '''returns the gradient at n'''
        return(-1 * (self.y[n] * self.data[n]) / (1 + math.exp(self.y[n] *
                                                  (self.w.dot(self.data[n])))))

    def runRegress(self, step):
        ''' run the regression for the learning rate = step'''

        first = True
        while(LA.norm(self.w - self.prev) > 0.01 or first):
            self.prev = self.w
            first = False
            permute = np.random.permutation(100)
            for i in permute:
                self.w = self.w - (step * self.getGrad(i))
            self.epoch += 1
        # print(self.w)
        return self.epoch

    def getEOut(self, n):
        ''' get the cross entropy error by generating n new points'''
        newData = np.array([(1.0, np.random.uniform(-1.0, 1.0),
                             np.random.uniform(-1.0, 1.0)) for i in range(n)])

        newY = np.array([int(np.sign(self.target.transpose().dot(x)))
                         for x in newData])

        error = 0
        for i in range(n):
            error += math.log(1 + (math.exp(-1 * newY[i] *
                                   self.w.dot(newData[i]))))

        error = error / (n)
        return error


def getAverage(times, points, step, newPoints):
    ''' Get the average number of epochs for n = times runs of the regression
        with N = points. ALso gets the average Eout.
    '''
    epoch = 0
    error = 0
    for i in range(times):
        lin = LogRegress(points)
        epoch += lin.runRegress(step)
        error += lin.getEOut(newPoints)
    epoch = epoch // times
    error /= times
    print("Average epoch:")
    print(epoch)
    print("Average Eout:")
    print(error)


getAverage(100, 100, 0.01, 1000)









