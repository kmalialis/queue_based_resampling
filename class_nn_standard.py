# -*- coding: utf-8 -*-

from keras.models import Model
from class_nn_base import NN_base
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization

###########################################################################################
#                                     Standard NN class                                   #
###########################################################################################

class NN_standard(NN_base):

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            layer_dims,
            learning_rate,
            output_activation,
            loss_function,
            weight_init,
            num_epochs,
            class_weights,
            minibatch_size,
            L2_lambda=0.0,
            flag_batchnorm=False,
            seed=0
    ):

        NN_base.__init__(
            self,
            layer_dims=layer_dims,
            learning_rate=learning_rate,
            output_activation=output_activation,
            loss_function=loss_function,
            weight_init=weight_init,
            num_epochs=num_epochs,
            class_weights=class_weights,
            minibatch_size=minibatch_size,
            L2_lambda=L2_lambda,
            flag_batchnorm=flag_batchnorm,
            seed=seed
        )

        # model
        self.model = self.create_standard_model()

        # configure model for training
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.loss_function,
            metrics=['accuracy']
        )

    ##################
    # Standard Model #
    ##################

    def create_standard_model(self):
        # Input and output dims
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]

        # Input layer
        X_input = Input(shape=(n_x,), name='input')

        #  First hidden layer
        X = Dense(
            units=self.layer_dims[1],
            activation=None,
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=l2(self.L2_lambda),
            bias_regularizer=None,
            activity_regularizer=None
        )(X_input)
        if self.flag_batchnorm:
            X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.01)(X)

        #  Other hidden layers (if any)
        for l in self.layer_dims[2:-1]:
            X = Dense(
                units=l,
                activation=None,
                use_bias=True,
                kernel_initializer=self.weight_init,
                bias_initializer='zeros',
                kernel_regularizer=l2(self.L2_lambda),
                bias_regularizer=None,
                activity_regularizer=None
            )(X)
            if self.flag_batchnorm:
                X = BatchNormalization()(X)
            X = LeakyReLU(alpha=0.01)(X)

        # Output layer
        y_out = Dense(
            units=n_y,
            activation=self.output_activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=l2(self.L2_lambda),
            bias_regularizer=None,
            activity_regularizer=None,
            name='output'
        )(X)

        # Model
        return Model(inputs=X_input, outputs=y_out)
