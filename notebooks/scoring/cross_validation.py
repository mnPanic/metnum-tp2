import time
import numpy as np
import statistics
import threading

from typing import Dict
from typing import Callable

ScoringFunc = Callable[[np.ndarray, np.ndarray], float]

def cross_validate(
        clf, 
        X, y, 
        scoring: ScoringFunc, 
        K: int,
        debug=False, 
        **kwargs
    ) -> float:
    """"
    Hace K fold del classifier `clf`, llamando a la funcion de scoring
    para calcular los scores de cada fold.
    """

    scores = []
    
    set_size = int(X.shape[0]/K)
    
    for i in range(K):
        # Particionar
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        X_train = np.block([[X[:l_bound]],[X[r_bound:]]])
        y_train = np.block([[y[:l_bound]],[y[r_bound:]]])
        X_val = X[l_bound:r_bound]
        y_val = y[l_bound:r_bound]

        # Entrenar
        clf.fit(X_train, y_train.ravel())
          
        # Clasificar
        y_pred = clf.predict(X_val, **kwargs)

        # Scoring
        scores.append(scoring(y_val, y_pred))
    
    if debug: print("scores:", scores)

    return statistics.mean(scores)

def cross_validate_fns(
        clf, 
        X, y, 
        scoring_fns: Dict[str, ScoringFunc],
        K: int,
        debug=False, 
        **kwargs
    ) -> float:
    """"
    Hace K fold del classifier `clf`, llamando a cada funcion de scoring y
    devolviendo los resultados bajo las mismas claves.
    No toma timings.
    """

    # Para guardar los scores de cada funcion provista
    scorings = {key:[] for key in scoring_fns.keys()}
    
    set_size = int(X.shape[0]/K)
    
    for i in range(K):
        # Particion
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        X_train = np.block([[X[:l_bound]],[X[r_bound:]]])
        y_train = np.block([[y[:l_bound]],[y[r_bound:]]])
        X_val = X[l_bound:r_bound]
        y_val = y[l_bound:r_bound]

        # Entrenamiento
        clf.fit(X_train, y_train.ravel())
          
        # Clasificacion
        y_pred = clf.predict(X_val, **kwargs)

        # Calculo de scores
        for name, scoring in scoring_fns.items():
            scorings[name].append(scoring(y_val, y_pred))
    
    if debug: print("scores:", scorings)

    # Calcular los mean scores
    mean_scores = dict.fromkeys(scorings, 0.0)

    for name, scores in scorings.items():
        mean_scores[name] = statistics.mean(scores)

    return mean_scores

def cross_validate_concurrent(
        clf, 
        X_dataset, 
        y_dataset, 
        scoring, 
        K, 
        debug=False, 
        **kwargs
    ) -> float:
    """"
    Hace K fold del classifier `clf`, llamando a la funcion de scoring
    para calcular los scores de cada fold.
    Lanza un thread por fold
    """
    dict_scores = {
        "score":      [],
        "score_time": [],
        "fit_time":   [],
    }
    
    threads = []
    set_size = int( (1.0/K) * X_dataset.shape[0])
    
    for i in range(K):
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        X_train = np.block([[X_dataset[:l_bound]],[X_dataset[r_bound:]]])
        y_train = np.block([[y_dataset[:l_bound]],[y_dataset[r_bound:]]])
        X_val = X_dataset[l_bound:r_bound]
        y_val = y_dataset[l_bound:r_bound]

        t = threading.Thread(
            target=fold, 
            kwargs={
                **{
                    "X_train": X_train, "y_train": y_train,
                    "X_val":   X_val,   "y_val":   y_val,
                    "clf": clf,
                    "scoring": scoring,
                    "scores": dict_scores,
                    "i": i,
                    "debug": debug,
                }, **kwargs
            },
        )

        t.start()
        threads.append(t)

    # Esperamos que terminen todos
    for thread in threads:
        thread.join()

    if debug: print("scores:", dict_scores)
    return statistics.mean(dict_scores["score"])

def fold(
        clf,
        X_train, y_train,
        X_val, y_val,
        scoring,
        scores,
        i,
        debug: bool,
        **kwargs
    ):
    if debug: print(f"running {i}")
    start_time = time.time()
    clf.fit(X_train, y_train.ravel())
    fit_time_elapsed = time.time() - start_time
        
    start_time = time.time()
    y_pred = clf.predict(X_val, **kwargs)
    score = scoring(y_val, y_pred)
    score_time_elapsed = time.time() - start_time

    scores["score"].append(score)
    scores["score_time"].append(score_time_elapsed)
    scores["fit_time"].append(fit_time_elapsed)
    if debug: print(f"finished {i}")


def cross_validate_pca(
        clf, 
        X, y, 
        scoring: ScoringFunc, 
        K: int,
        debug=False, 
        **kwargs
    ) -> float:
    """"
    Hace K fold del classifier `clf`, llamando a la funcion de scoring
    para calcular los scores de cada fold.
    """

    scores = []
    
    set_size = int(X.shape[0]/K)
    
    for i in range(K):
        # Particionar
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        X_train = np.block([[X[:l_bound]],[X[r_bound:]]])
        y_train = np.block([[y[:l_bound]],[y[r_bound:]]])
        X_val = X[l_bound:r_bound]
        y_val = y[l_bound:r_bound]

        # Entrenar
        clf.fit(X_train, y_train.ravel(),i)
          
        # Clasificar
        y_pred = clf.predict(X_val, **kwargs)

        # Scoring
        scores.append(scoring(y_val, y_pred))
    
    if debug: print("scores:", scores)

    return statistics.mean(scores)
