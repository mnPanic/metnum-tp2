import sklearn.metrics as metrics

def precision_score(y_true, y_pred):
    """
    Calcula la precision, i.e los aciertos relativos dentro de una clase. 
    Dada una clase i,

        prec(i) = tp_i / (tp_i + fp_i)

    La precision en el caso de un clasificador de muchas clases, se
    define como el promedio de las precision para cada una de las
    clases.
    Simple wrapper de sklearn.metrics.precision_score
    """
    return metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)

def accuracy_score(y_true, y_pred):
    """Calcula el accuracy, los aciertos totales sobre los casos totales."""
    return metrics.accuracy_score(y_true, y_pred)

def recall_score(y_true, y_pred):
    """
    Calcula el recall, una metrica para medir los reconocimientos dentro de una
    clase. Dada una clase i,
    
        recall(i) = tpi / (tp_i + fn_i)

    """
    return metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)

def f1_score(y_true, y_pred):
    """
    Calcula F1. Dado que precision y recall son dos medidas
    importantes que no necesariamente tienen la misma calidad
    para un mismo clasificador, se define la esta metrica F1 para medir
    un compromiso entre el recall y la precision.
    Se define como

        2 ∗ precision ∗ recall/(precision + recall)

    """
    return metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)