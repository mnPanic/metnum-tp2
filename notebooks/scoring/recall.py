import sklearn.metrics as metrics

def recall_score(y_true, y_pred):
    
    return metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)