#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides the ValueModel class, for defining a neural network model of the value function."""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# def reload_model_new(model_dir, inputshape):
#     """Load a model.

#     Args:
#         model_dir (str):
#             path to the model
#         inputshape (ndarray):
#             shape of the neural network input given by :attr:`otto.classes.sourcetracking.SourceTracking.NN_input_shape`
#     """
#     model_name = os.path.basename(model_dir)
#     weights_path = os.path.abspath(os.path.join(model_dir, model_name))
#     config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
#     with open(config_path, 'rb') as filehandle:
#         config = pickle.load(filehandle)
#     model = ValueModel_new(**config)
#     # model=model.cuda()
#     # model.build_graph(input_shape_nobatch=inputshape)
#     # model.load_weights(weights_path)
#     return model

def reload_model_new(model_dir, inputshape):

    model_name = os.path.basename(model_dir)
    config_path = os.path.join(model_dir, model_name + ".config")
    with open(config_path, 'rb') as filehandle:
        config = pickle.load(filehandle)

    # Recreate the model using the saved configuration
    model = ValueModel_new(**config)

    # Load the weights into the model
    weights_path = os.path.join(model_dir, model_name + ".pth")
    model.load_state_dict(torch.load(weights_path))

    return model


class ValueModel_new(nn.Module):
    """Neural network model used to predict the value of the belief state
    (i.e. the expected remaining time to find the source).

    Args:
        Ndim (int):
            number of space dimensions (1D, 2D, ...) for the search problem
        FC_layers (int):
            number of hidden layers
        FC_units (int or tuple(int)):
            units per layer
        regularization_factor (float, optional):
            factor for regularization losses (default=0.0)
        loss_function (str, optional):
            either 'mean_absolute_error', 'mean_absolute_percentage_error' or 'mean_squared_error' (default)

    Attributes:
        config (dict):
            saves the args in a dictionary, can be used to recreate the model

    """

    def __init__(self,
                 Ndim,
                 FC_layers,
                 FC_units,
                 inputshape,
                 regularization_factor=0.0,
                 loss_function='mean_squared_error',
                 ):
        """Constructor.


        """
        super(ValueModel_new, self).__init__()
        self.config = {"Ndim": Ndim,
                       "FC_layers": FC_layers,
                       "FC_units": FC_units,
                       "inputshape": inputshape,
                       "regularization_factor": regularization_factor,
                       "loss_function": loss_function,
                       }

        self.Ndim = Ndim
        self.FC_layers = FC_layers

        self.regularization_factor = regularization_factor
        
        self.loss_fxn = nn.MSELoss()

        # regularizer = regularizers.l2(regularization_factor)

        # # flattening
        # self.flatten = nn.Flatten()

        # # fully connected layers
        # self.FC_block = None
        in_features = torch.prod(torch.tensor(inputshape)).item()
        print("inputshape",inputshape)
        print("in_features",in_features)

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(in_features, FC_units),
        #     nn.ReLU(),
        #     nn.Linear(FC_units, FC_units),
        #     nn.ReLU(),
        #     nn.Linear(FC_units, FC_units),
        #     nn.ReLU(),
        #     nn.Linear(FC_units, FC_units),
        #     nn.ReLU(),
        #     nn.Linear(FC_units, 1)
        # )

        # self.relu = nn.ReLU()

        # Here, I have defined the layers for the Neural Network :

        self.layers = nn.ModuleList()  # Container for all layers
        # Input layer
        self.layers.append(nn.Linear(in_features, FC_units))
        # Hidden layers
        for _ in range(FC_layers):
            self.layers.append(nn.Linear(FC_units, FC_units))
        # Output layer
        self.layers.append(nn.Linear(FC_units, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
                
        # self._initialize_weights()

        if loss_function == 'mean_absolute_error':
            self.loss_function = nn.L1Loss(reduction="none")
        # elif loss_function == 'mean_absolute_percentage_error':
        #     self.loss_function = tf.keras.losses.MeanAbsolutePercentageError(reduction="none")
        elif loss_function == 'mean_squared_error':
            self.loss_function = nn.MSELoss(reduction="none")
        else:
            raise Exception("This loss function has not been made available")


    # def _initialize_weights(self):
        # for module in self.modules():
            # if isinstance(module, nn.Linear):
                # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # if module.bias is not None:
                    # nn.init.zeros_(module.bias)

    def forward(self,x):
        # for i in range(self.FC_layers + 2):
        #     layer_name = f'dense{i+1}'
        #     layer = getattr(self, layer_name)
        #     # x = torch.from_numpy(x).float() 
        #     x = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        #     # x = x.to("cuda")
        #     # print("type", layer)
        #     if i<(self.FC_layers + 1):
        #         x = F.relu(layer(x))
        #     else :
        #         x = layer(x)


        # x = torch.tensor(x.numpy(), dtype=torch.float32)

        x = torch.flatten(x, start_dim=1)  # I am flattening all dims except the batch size

        for layer in self.layers[:-1]:  
            x = torch.relu(layer(x))    
        x = self.layers[-1](x)          

        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)

        return x

    def train_step(self, x, y, augment=False):
        """A training step.

        Args:
            x (tf.tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.tensor with shape=(batch_size, 1)): batch of target values

        Returns:
            loss (tf.tensor with shape=()): total loss
        """

        # Add symmetric duplicates
        if augment:
            shape = x.shape
            if self.Ndim == 1:
                Nsym = 2
                x = x[None, ...]
                _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
                x = torch.cat([x, _], dim=0)
                x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            elif self.Ndim == 2:
                Nsym = 8
                x = x[None, ...]
                _ = x.permute(0, 1, 3, 2)  # transposition
                x = torch.cat([x, _], dim=0)
                _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
                x = torch.cat([x, _], dim=0)
                _ = torch.flip(x, dim=[2, 3])  # symmetry: x -> -x, y -> -y
                x = torch.cat([x, _], dim=0)
                x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            elif self.Ndim == 3:
                Nsym = 48
                x = x[None, ...]
                t1 = x.permute(0, 1, 3, 2, 4)  # transposition x <-> y
                t2 = x.permute(0, 1, 4, 3, 2)  # transposition x <-> z
                t3 = x.permute(0, 1, 2, 4, 3)  # transposition y <-> z
                t4 = x.permute(0, 1, 4, 2, 3)  # transposition x <-> y, y <-> z
                t5 = x.permute(0, 1, 3, 4, 2)  # transposition x <-> z, y <-> x
                x = torch.cat([x, t1, t2, t3, t4, t5], dim=0)
                _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
                x = torch.cat([x, _], dim=0)
                _ = torch.flip(x, dim=[2, 3])  # symmetry: x -> -x, y -> -y
                x = torch.cat([x, _], dim=0)
                _ = torch.flip(x, dim=[2, 3, 4])  # symmetry: x -> -x, y -> -y, z -> -z
                x = torch.cat([x, _], dim=0)
                x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            else:
                raise Exception("augmentation with symmetric duplicates is not implemented for Ndim > 3")

            # repeat target
            y = y[None, ...]
            y = torch.repeat_interleave(y, Nsym, dim=0)
            y = torch.reshape(y, shape=tuple([Nsym * shape[0]] + [1]))

        # print("xtype",type(x))

        y_pred = self(x)
        loss = self.loss_fxn(y_pred, y)
        loss_reg = self.compute_l2_regularization()
        loss_tot = loss + loss_reg
        self.optimizer.zero_grad()
        loss_tot.backward()
        self.optimizer.step()

        return loss_tot.item()


    def compute_l2_regularization(self):
        l2_penalty = 0.0
        for name, param in self.named_parameters():
            # I did not apply l2 reg to biases
            if "weight" in name and param.requires_grad:
                l2_penalty += torch.sum(param ** 2)
        return self.regularization_factor * l2_penalty 

    # def call(self, x, training=False, sym_avg=False):
    #     """Call the value model

    #     Args:
    #         x (ndarray or tf.tensor with shape (batch_size, input_shape)):
    #             array containing a batch of inputs
    #         training (bool, optional):
    #             whether this call is done during training (as opposed to evaluation) (default=False)
    #         sym_avg (bool, optional):
    #             whether to take the average value of symmetric duplicates (default=False)

    #     Returns:
    #         x (tf.tensor with shape (batch_size, 1))
    #             array containing a batch of values
    #     """
    #     shape = x.shape  # (batch_size, input_shape)
    #     ensemble_sym_avg = False
    #     if sym_avg and (shape[0] is not None):
    #         ensemble_sym_avg = True

    #     # create symmetric duplicates
    #     if ensemble_sym_avg:
    #         if self.Ndim == 1:
    #             Nsym = 2
    #             x = x[None, ...]
    #             _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
    #             x = torch.cat([x, _], dim=0)
    #             x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
    #         elif self.Ndim == 2:
    #             Nsym = 8
    #             x = x[None, ...]
    #             _ = x.permute(0, 1, 3, 2)  # transposition
    #             x = torch.cat([x, _], dim=0)
    #             _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
    #             x = torch.cat([x, _], dim=0)
    #             _ = torch.flip(x, dim=[2, 3])  # symmetry: x -> -x, y -> -y
    #             x = torch.cat([x, _], dim=0)
    #             x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
    #         elif self.Ndim == 3:
    #             Nsym = 48
    #             x = x[None, ...]
    #             t1 = x.permute(0, 1, 3, 2, 4)  # transposition x <-> y
    #             t2 = x.permute(0, 1, 4, 3, 2)  # transposition x <-> z
    #             t3 = x.permute(0, 1, 2, 4, 3)  # transposition y <-> z
    #             t4 = x.permute(0, 1, 4, 2, 3)  # transposition x <-> y, y <-> z
    #             t5 = x.permute(0, 1, 3, 4, 2)  # transposition x <-> z, y <-> x
    #             x = torch.cat([x, t1, t2, t3, t4, t5], dim=0)
    #             _ = torch.flip(x, dim=[2])  # symmetry: x -> -x
    #             x = torch.cat([x, _], dim=0)
    #             _ = torch.flip(x, dim=[2, 3])  # symmetry: x -> -x, y -> -y
    #             x = torch.cat([x, _], dim=0)
    #             _ = torch.flip(x, dim=[2, 3, 4])  # symmetry: x -> -x, y -> -y, z -> -z
    #             x = torch.cat([x, _], dim=0)
    #             x = torch.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
    #         else:
    #             raise Exception("symmetric duplicates for Ndim > 3 is not implemented")

    #     # flatten input
    #     x = self.flatten(x)

    #     # reduce the symmetric outputs
    #     if ensemble_sym_avg:
    #         x = torch.reshape(x, shape=(Nsym, shape[0], 1))
    #         x = torch.mean(x, dim=0)
                 

    #     # New forward pass
    #     x = self.forward(x)

    #     # # forward pass
    #     # if self.FC_block is not None:
    #     #     for i in range(len(self.FC_block)):
    #     #         x = self.FC_block[i](x, training=training)

    #     # x = self.densefinal(x)

    #     return x  # (batch_size, 1)


    # def build_graph(self, input_shape_nobatch):
    #     """Builds the model. Use this function instead of model.build() so that a call to
    #     model.summary() gives shape information.

    #     Args:
    #         input_shape_nobatch (tuple(int)):
    #             shape of the neural network input given by :attr:`otto.classes.sourcetracking.SourceTracking.NN_input_shape`
    #     """
    #     input_shape_nobatch = tuple(input_shape_nobatch)
    #     input_shape_withbatch = tuple([1] + list(input_shape_nobatch))
    #     self.build(input_shape_withbatch)
    #     inputs = tf.keras.Input(shape=input_shape_nobatch)
    #     _ = self.call(inputs)

    # note: the tf.function decorator prevent using tensor.numpy() for performance reasons, use only tf operations
    # @tf.function


    # @tf.function_WorkerBatch
    def test_step(self, x, y):
        """ A test step.

        Args:
            x (tf.tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.tensor with shape=(batch_size, 1)): batch of target values

        Returns:
            loss (tf.tensor with shape=()): total loss
        """

        with torch.no_grad():
            y_pred = self(x)
            loss = self.loss_fxn(y_pred, y)
            loss_reg = self.compute_l2_regularization()
            loss_tot = loss + loss_reg
        return loss_tot.item()


    def save_model(self, model_dir):

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        model_name = os.path.basename(model_dir)
        weights_path = os.path.join(model_dir, model_name + ".pth")
        torch.save(self.state_dict(), weights_path)

        config_path = os.path.join(model_dir, model_name + ".config")
        with open(config_path, 'wb') as filehandle:
            pickle.dump(self.config, filehandle)

        # model_name = os.path.basename(model_dir)
        # weights_path = os.path.abspath(os.path.join(model_dir, model_name))
        # self.save_weights(weights_path, save_format='h5')
        # config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
        # with open(config_path, 'wb') as filehandle:
        #     pickle.dump(self.config, filehandle)
