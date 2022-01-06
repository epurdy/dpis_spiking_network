"""This file is incomplete, and is not tested at all. It probably
contains serious bugs, and it is definitely missing core
functionality.

This file is theoretically going to evolve into a feature-complete
version that is clean enough to build on.

This file not working, or being extremely slow, should not be held
against Dpi's proposal.
"""

import numpy as np
import tensorflow as tf

class SONN:
    def __init__(self, *, nsyn, nsize, batch_size, num_classes, num_cycles,
                 input_size, thresholds, threshold):
        self.num_cycles = num_cycles
        self.nsyn = nsyn
        self.nsize = nsize
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.from_ = np.zeros((num_classes, nsize, nsyn), dtype=np.int16)
        self.weights = np.zeros((num_classes, nsize, nsyn), dtype=np.int64)
        self.thresholds = thresholds
        self.threshold=threshold

        self.quant_pos = 0.0
        self.quant_div_pos = 100.0
        self.quant_up_pos = 19.0
        self.quant_neg = 0.0
        self.quant_div_neg = 900.0
        self.quant_up_neg = 150.0
        
        self.num_new = 10
        self.div_new = 10

        self.adj_pos = 500
        self.adj_neg = 100
        
    def get_num_spikes(self, *, x, y):
        # [batch, classes]
        num_spikes = np.zeros((len(y), self.num_classes), dtype=np.int64)
        for i in range(x.shape[0]):
            # [classes, nsize]
            potential = np.zeros((self.num_classes, self.nsize), dtype=np.int64)
            for time in range(self.num_cycles):            
                potential += self.weights[x[i, self.from_] > self.thresholds[time]].sum(axis=-1)
                # [classes, nsize]
                spike = (potential > self.threshold)
                num_spikes[i] += spike.sum(axis=-1)
                potential[~spike] >>= 1
        return num_spikes
            
    def test(self, *, x, y, learning=False):
        # [batch, classes]
        num_spikes = self.get_num_spikes(x=x, y=y)
        # [batch]
        pred = num_spikes.argmax(axis=-1)
        # scalar
        num_hits = (pred == y).sum()

        if learning:
            # [batch]
            error = num_spikes.max(axis=-1) - np.diag(num_spikes[:, y])
            return num_hits / len(y), error

        return num_hits / len(y)

    def connect(self, *, num_new, x, y, init):
        if self.from_

    def learn(self, *, x, y, adj_pos):
        pass
    
    def train(self, *, x, y, error):
        for i in range(x.shape[0]):
            if error[i] >= self.quant_pos:
                self.connect(num_new=self.num_new, x=x[i], y=y[i],
                             init=self.threshold/self.div_new)
                self.learn(x=x[i], y=y[i], adj_pos=self.adj_pos)
                
    def train_test_loop(self, *, x_train, x_test, y_train, y_test):
        for epoch in range(1_000_000_000):
            training_indices = list(range(len(x_train)))
            np.random.shuffle(training_indices)
            for batch_start in range(0, len(training_indices), self.batch_size):
                batch = training_indices[batch_start:batch_start + self.batch_size]

                accuracy, error = self.test(x=x_train[batch],
                                            y=y_train[batch],
                                            learning=True)
                self.train(x=x_train[batch], y=y_train[batch], error=error)

            print('end of epoch', epoch)
            print('\ttrain', self.test(x=x_train, y=y_train))
            print('\ttest', self.test(x=x_train, y=y_train))
                    
            

        
def main(
        nsyn=10,
        nsize=792,
        threshold=1_000_000,
        threshold_min = 300_000,
        threshold_max = 1_200_000,
        min_weight = 97_000,
        num_cycles=4,
        color_thresholds=(192, 128, 64, 1),
        seed=1000,
        batch_size=2,
):
    np.random.seed(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_classes = 1 + max(y_train.max(), y_test.max())
    # reshape image axes
    assert len(x_train.shape) == 3
    assert len(x_test.shape) == 3
    x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
    
    sonn = SONN(nsyn=nsyn, nsize=nsize, batch_size=batch_size,
                num_classes=num_classes, input_size=x_train.shape[1],
                num_cycles=num_cycles, thresholds=color_thresholds,
                threshold=threshold)

    sonn.train_test_loop(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test)



if __name__ == '__main__':
    main()
