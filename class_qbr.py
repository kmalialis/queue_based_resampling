# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from class_sota import Baseline

###########################################################################################
#                                Queue-Based Resampling (QBR)                             #
###########################################################################################


class QBR(Baseline):

    ###############
    # Constructor #
    ###############

    def __init__(self, model, queue_size_budget):
        Baseline.__init__(self, model)

        # budget
        self.budget = queue_size_budget

        # init queues
        self.xs_neg = deque(maxlen=1)
        self.ys_neg = deque(maxlen=1)

        self.xs_pos = deque(maxlen=1)
        self.ys_pos = deque(maxlen=1)

    #############
    # Auxiliary #
    #############

    def adapt_queue(self, q, q_cap):
        if q == 'neg':
            self.xs_neg = deque(self.xs_neg, q_cap)
            self.ys_neg = deque(self.ys_neg, q_cap)
        elif q == 'pos':
            self.xs_pos = deque(self.xs_pos, q_cap)
            self.ys_pos = deque(self.ys_pos, q_cap)

    #######
    # API #
    #######

    def get_training_set(self, n_features):
        # merge queues
        xs = list(self.xs_neg) + list(self.xs_pos)
        ys = list(self.ys_neg) + list(self.ys_pos)

        # convert merged queues to np arrays
        size = len(ys)  # current queue size
        x = np.array(xs).reshape(size, n_features)
        y = np.array(ys).reshape(size, 1)

        #  batch GD
        self.model.change_minibatch_size(size)

        # return
        return x, y

    def append_to_queues(self, x, y):
        if y == 0:
            self.xs_neg.append(x)
            self.ys_neg.append(y)

            length = len(self.ys_neg)
            capacity = self.ys_neg.maxlen
            if length == capacity and capacity < self.budget / 2.0:
                self.adapt_queue('neg', capacity + 1)
        else:
            self.xs_pos.append(x)
            self.ys_pos.append(y)

            length = len(self.ys_pos)
            capacity = self.ys_pos.maxlen
            if length == capacity and capacity < self.budget / 2.0:
                self.adapt_queue('pos', capacity + 1)
