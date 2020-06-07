import sklearn.metrics as metrics

def f1_score(y_true, y_pred):
    
    return metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)