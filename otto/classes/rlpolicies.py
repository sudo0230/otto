#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from copy import deepcopy
import tensorflow as tf

# _____________________  parameters  _____________________
EPSILON = 1e-10
EPSILON_CHOICE = EPSILON
# ________________________________________________________


class RLPolicies:
    def __init__(
            self,
            env,
            model,
            sym_avg,
    ):
        self.env = env
        self.model = model
        self.sym_avg = sym_avg
        self.policy = -1
        self.policy_name = "RL"

    def choose_action(self, ):

        inputs = [0] * self.env.Nactions
        probs = [0] * self.env.Nactions
        ishape = self.env._inputshape()

        for action in range(self.env.Nactions):
            inputs[action] = []
            probs[action] = []

            # moving agent
            agent_, move_possible = self.env._move(action, self.env.agent)

            # extracting the evidence matrix for Bayesian inference
            p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

            # updating p_source by Bayesian inference
            p_source_ = deepcopy(self.env.p_source)
            p_source_ = p_source_[np.newaxis, :] * p_evidence

            for h in range(self.env.Nhits):
                prob = np.sum(p_source_[h])  # norm is the proba to transit to this state = (1-pend) * p(hit)
                prob = np.maximum(EPSILON, prob)
                input = self.env._centeragent(p_source_[h], agent_)
                probs[action].append(prob)
                inputs[action].append(input)

        inputs = np.asarray(inputs, dtype=np.float32)  # (Nactions, Nhits, inputshape)
        probs = np.asarray(probs, dtype=np.float32)  # (Nactions, Nhits)
        assert inputs.shape == tuple([self.env.Nactions] + [self.env.Nhits] + list(ishape))
        assert probs.shape == tuple([self.env.Nactions] + [self.env.Nhits])

        inputs_shape = inputs.shape
        inputs = tf.reshape(inputs, shape=tuple([inputs_shape[0] * inputs_shape[1]] + list(inputs_shape[2:])))  # (Nactions * Nhits, inputshape)
        assert inputs.shape == tuple([self.env.Nactions * self.env.Nhits] + list(ishape))

        values_next = self.model(inputs, sym_avg=self.sym_avg)  # (Nactions*Nhits, 1)
        assert values_next.shape == tuple([self.env.Nactions * self.env.Nhits] + [1])

        values_next = tf.reshape(values_next, shape=(inputs_shape[0], inputs_shape[1],))  # (Nactions, Nhits)
        assert values_next.shape == tuple([self.env.Nactions] + [self.env.Nhits])
        assert values_next.shape == probs.shape

        expected_value = 1.0 + tf.math.reduce_sum(probs * values_next, axis=1)  # sum over hits: (Nactions)
        assert expected_value.shape == tuple([self.env.Nactions])

        # TODO check shapes
        # action_chosen = np.argmin(expected_value.numpy().squeeze())

        action_chosen = np.argwhere(np.abs(expected_value - np.min(expected_value)) < EPSILON_CHOICE).flatten()[0]
        return action_chosen


