import time
import numpy as np
import statistics
import threading

def cross_validate(
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
    """

    dict_scores = {
        "score":      [],
        "score_time": [],
        "fit_time":   [],
    }
    
    set_size = int( (1.0/K) * X_dataset.shape[0])
    
    for i in range(K):
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        start_time = time.time()
        X_train = np.block([[X_dataset[:l_bound]],[X_dataset[r_bound:]]])
        y_train = np.block([[y_dataset[:l_bound]],[y_dataset[r_bound:]]])
        X_val = X_dataset[l_bound:r_bound]
        y_val = y_dataset[l_bound:r_bound]
        
        clf.fit(X_train, y_train.ravel())
        end_time = time.time()
        fit_time_elapsed = end_time - start_time
          
        start_time = time.time()
        y_pred = clf.predict(X_val, **kwargs)
        score = scoring(y_val, y_pred)
        end_time = time.time()
        score_time_elapsed = end_time - start_time

        dict_scores["score"].append(score)
        dict_scores["score_time"].append(score_time_elapsed)
        dict_scores["fit_time"].append(fit_time_elapsed)
    
    if debug: print("scores:", dict_scores)

    return statistics.mean(dict_scores["score"])

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

