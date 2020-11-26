from tensorflow.keras import backend as K

# Calculates true positive, true negative, false postive, and false negative'
def __binary_metrics(y_true, y_pred):

    # Calculating the correct reults
    TP = K.sum(y_true * y_pred)
    TN = K.sum((1-y_true) * (1-y_pred))

    # Calculating the incorrect results
    FN = K.sum(y_true * (1-y_pred))
    FP = K.sum((1-y_true) * y_pred)

    return TP, FP, TN, FN


def sensitivity(y_true, y_pred):

    ''' True positive rate'''

    TP, _, _, FN = __binary_metrics(y_true, y_pred)
    return TP / (TP + FN +  K.epsilon())


def sepcificity(y_true, y_pred):

    ''' True negative rate '''

    _, FP, TN, _ = __binary_metrics(y_true, y_pred)
    return TN / (TN + FP +  K.epsilon())


def precision(y_true, y_pred):

    ''' Positive predictive value '''

    TP, FP, _, _ = __binary_metrics(y_true, y_pred)
    return TP / (TP + FP +  K.epsilon())


def negative_prediction(y_true, y_pred):

    ''' Negative predictive value '''

    _, _, TN, FN = __binary_metrics(y_true, y_pred)
    return TN / (TN + FN +  K.epsilon())


def miss_rate(y_true, y_pred):

    ''' Negative predictive value '''

    TP, _, _, FN = __binary_metrics(y_true, y_pred)
    return FN / (FN + TP +  K.epsilon())


def fall_out(y_true, y_pred):

    ''' False positive rate '''

    _, FP, TN, _ = __binary_metrics(y_true, y_pred)
    return FP / (FP + TN +  K.epsilon())


def false_discovery(y_true, y_pred):

    ''' False discovery rate '''

    TP, FP, _, _ = __binary_metrics(y_true, y_pred)
    return FP / (FP + TP +  K.epsilon())


def false_omission(y_true, y_pred):

    ''' False omission rate '''

    _, _, TN, FN  = __binary_metrics(y_true, y_pred)
    return FN / (FN + TN +  K.epsilon())
