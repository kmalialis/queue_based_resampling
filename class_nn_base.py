# -*- coding: utf-8 -*-

import numpy as np
from keras.initializers import glorot_uniform, he_normal

###########################################################################################
#                                       NN class                                          #
###########################################################################################

class NN_base():

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            layer_dims,                 # [n_x, n_h1, n_h2, .., n_hL, n_y]
            learning_rate,
            output_activation,
            loss_function,
            weight_init,
            num_epochs,
            class_weights,
            minibatch_size,
            L2_lambda,
            flag_batchnorm,
            seed
    ):

        # seed
        self.seed = seed

        # NN parameters
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.num_epochs = num_epochs
        self.class_weights = class_weights
        self.minibatch_size = minibatch_size
        self.L2_lambda = L2_lambda
        self.flag_batchnorm = flag_batchnorm

        if weight_init == "glorot":
            self.weight_init = glorot_uniform(seed=self.seed)
        elif weight_init == "he":
            self.weight_init = he_normal(seed=self.seed)

        # loss function
        self.loss_function = loss_function

        # model to be defined in sub-classes
        self.model = None

    #############
    # Auxiliary #
    #############

    def cast_classes(self, y_datasets):
        return [y_data.astype('int') for y_data in y_datasets]

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ##############
    # Prediction #
    ##############

    def prediction(self, x):
        y_hat = self.model.predict(x=x, verbose=0)
        y_hat_class = np.around(y_hat)

        return y_hat, y_hat_class

    ############
    # Training #
    ############

    def training(self, x, y):
        self.model.fit(
            x=x,
            y=self.cast_classes([y]),           # cast class to integer
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            class_weight=self.class_weights,
            verbose=0                           # 0: off, 1: full, 2: brief
        )

    ########################
    # Change class weights #
    ########################

    def change_class_weights(self, weights):
        if self.class_weights != weights:
            self.class_weights = weights

    ##################
    # Get num_epochs #
    ##################

    def get_num_epochs(self):
        return self.num_epochs

    #####################
    # Change num_epochs #
    #####################

    def change_num_epochs(self, n_epochs):
        if self.num_epochs != n_epochs:
            self.num_epochs = n_epochs

    #########################
    # Change minibatch size #
    #########################

    def change_minibatch_size(self, batch_size):
        if self.minibatch_size != batch_size:
            self.minibatch_size = batch_size
