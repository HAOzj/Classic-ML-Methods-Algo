# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on JUL 5, 2019

@author: woshihaozhaojun@sina.com
"""
import numpy as np


def mean_absolute_percent_error(y_true, y_pred, sample_weight=None):
    """计算平均百分比绝对误差.

    要保证真实值中没有0!

    Args:
         y_true(array-like) :-  of shape (n_samples), 真实值的列表或者序列
         y_pred(array-like) :-  of shape (n_samples), 预测值的列表或者序列
         sample_weight(array-like) :-  of shape (n_samples), 样例权重的列表或者序列
    Returns:
        float, 预测值相对真实值的平均百分比绝对误差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true.reshape(-1,)
    y_pred.reshape(-1,)
    
    try: 
        assert all(y_true != 0)
    except AssertionError:
        raise ValueError("y_true中不能有0")

    try:
        assert y_true.shape == y_pred.shape
    except AssertionError:
        raise ValueError("y_true和y_pred的维度必需一致")
        
    if not sample_weight:
        sample_weight = np.array([1 for _ in range(y_true.shape[0])])

    sample_weight = np.array(sample_weight)
    return np.mean(abs((y_pred - y_true) / y_true * sample_weight))
