# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on 21 FEB, 2020

@author: woshihaozhaojun@sina.com
"""
import torch
import copy
import numpy as np


def dropout_strict(w, keep_prob):
    """implement inverted dropout ensuring that the share of kept neurons is strictly keep_prob.

    Args:
        w (torch.tensor) : weights before dropout
        keep_prob(float) : keep probability
    """
    k = round(w.shape[1] * keep_prob)
    _, indices = torch.topk(torch.randn(w.shape[0], w.shape[1]), k)
    keep = torch.zeros(4, 4).scatter_(dim=1, index=indices, src=torch.ones_like(w))
    w *= keep
    w /= keep_prob


def dropout_loose(w, keep_prob):
    """A simple Implementation of inverted dropout.

    Args:
        w(np.array) :- neurons subject to dropout
        keep_prob(float) :- keep probability
    """
    keep = np.random.rand(w.shape[0], w.shape[1]) < keep_prob
    w *= keep
    w /= keep_prob
