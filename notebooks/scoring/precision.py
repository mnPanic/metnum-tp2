import sklearn.metrics as metrics

def precision_score(y_true, y_pred):
    
    return metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
    