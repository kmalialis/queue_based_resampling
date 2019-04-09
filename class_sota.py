# -*- coding: utf-8 -*-

import numpy as np
from collections import deque

###########################################################################################
#                                        Baseline                                         #
###########################################################################################


class Baseline:
    # Constructor
    def __init__(self, model):
        self.model = model

    # predict
    def predict(self, x):
        return self.model.prediction(x)

    # train
    def train(self, x, y):
        self.model.training(x, y)

###########################################################################################
#                              (Adaptive) Cost Sensitive Learning                         #
###########################################################################################


class CS(Baseline):
    def __init__(self, model, update_freq, upper_weight):
        Baseline.__init__(self, model)

        # default values from paper
        self.cs_weight_abnormal = 0.05  # abnormal
        self.cs_weight_normal = 0.95  # normal
        self.cs_weight_pos = self.cs_weight_normal / self.cs_weight_abnormal
        self.cs_weight_neg = 1.0

        # adaptive cs
        self.update_freq = update_freq
        self.upper_weight = upper_weight

        # init
        class_weights = {0: self.cs_weight_neg, 1: self.cs_weight_pos}
        self.model.change_class_weights(class_weights)

    def adapt_costs(self, current_time, new_cs_weight_neg, new_cs_weight_pos):
        if current_time % self.update_freq == 0:
            if new_cs_weight_neg > self.upper_weight:
                new_cs_weight_neg = self.upper_weight

            if new_cs_weight_pos > self.upper_weight:
                new_cs_weight_pos = self.upper_weight

            class_weights = {0: new_cs_weight_neg, 1: new_cs_weight_pos}
            self.model.change_class_weights(class_weights)

###########################################################################################
#                                       Sliding Window                                    #
###########################################################################################


class Sliding(Baseline):
    def __init__(self, model, sliding_window_size):
        Baseline.__init__(self, model)

        # init
        self.sliding_window_size = sliding_window_size

        self.xs = deque(maxlen=self.sliding_window_size)
        self.ys = deque(maxlen=self.sliding_window_size)

    # Add to / remove from sliding window
    def append_to_win(self, x, y, n_features):
        # Append to queues (sliding windows)
        self.xs.append(x)
        self.ys.append(y)

        # batch GD
        size = len(self.ys)
        self.model.change_minibatch_size(size)

        # convert queues to np arrays
        x = np.array(self.xs).reshape(size, n_features)
        y = np.array(self.ys).reshape(size, 1)

        # return
        return x, y

###########################################################################################
#                                              OOB                                        #
###########################################################################################

class OOB():
    # Constructor
    def __init__(self, models):
        self.models = models
        self.flags_train = np.zeros(len(self.models))

    # predict
    def predict(self, x):
        preds = [m.prediction(x)[0] for m in self.models]
        y_hats = [a.flatten()[0] for a in preds]
        y_hats_avg = np.mean(y_hats).reshape(1, 1)
        y_hats_avg_class = np.around(y_hats_avg)

        return y_hats_avg, y_hats_avg_class

    # train
    def train(self, x, y):
        for m in self.models:
            if m.get_num_epochs() != 0:
                m.training(x, y)

    # OOB
    def oob_oversample(self, random_state, imbalance_rate):
        for m in self.models:
            # sample from Poisson
            k = random_state.poisson(imbalance_rate)

            # change number of epochs
            m.change_num_epochs(k)
