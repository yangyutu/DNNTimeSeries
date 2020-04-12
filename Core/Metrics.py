
import numpy as np
import torch


def convertToNumpy(x, y):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy(), y.cpu().numpy()
    return x, y


def RSE(y_pred, y_true):
    '''root squared sum difference divided by the standard dievation of y_true'''
    y_pred, y_true = convertToNumpy(y_pred, y_true)
    rse = np.sqrt(np.square(y_pred - y_true).sum()) / np.sqrt(np.square(y_true - y_true.mean()).sum())
    return rse

def QuantileLoss(y_true, y_pred, qs):
    '''
    Quantile loss version
    Args:
    y_true (batch_size, output_horizon)
    y_pred (batch_size, output_horizon, num_quantiles)
    '''
    y_pred, y_true = convertToNumpy(y_pred, y_true)

    L = np.zeros_like(y_true)
    for i, q in enumerate(qs):
        yq = y_pred[:, :, i]
        diff = yq - y_true
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SymMeanAPE(y_true, y_pred):
    '''Symmetric mean absolute percentage error'''
    y_pred, y_true = convertToNumpy(y_pred, y_true)

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel() + 1e-6
    mean_y = (y_true + y_pred) / 2.
    return np.mean(np.abs((y_true - y_pred) / mean_y))

def MeanAPE(y_true, y_pred):
    '''Mean absolute percentage error'''
    y_pred, y_true = convertToNumpy(y_pred, y_true)

    y_true = np.array(y_true).ravel() + 1e-6
    y_pred = np.array(y_pred).ravel()
    return np.mean(np.abs((y_true - y_pred) \
        / y_true))

def NormRMSE(y_true, y_pred):
    """
    Normalized Root Mean Square Error
    """
    y_pred, y_true = convertToNumpy(y_pred, y_true)

    mean = np.mean(np.abs(y_true)) + 1e-6
    error = np.square(np.subtract(y_pred, y_true)).mean()

    return error / mean

def NormDeviation(y_true, y_pred):
    """
    Normalized Deviation
    """
    convertToNumpy(y_true, y_pred)

    mean = np.mean(np.abs(y_true)) + 1e-6
    deviation = np.mean(np.abs(y_true - y_pred))

    return deviation / mean 

def MAE(y_true, y_pred):
    """
    Mean Absolute Error
    """
    convertToNumpy(y_true, y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

