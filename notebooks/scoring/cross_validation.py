import time
import numpy as np
import statistics 

def cross_validate(estimator, X_dataset, y_dataset, scoring, K):
    dict_scores = {}
    dict_scores["score"] = []
    dict_scores["score_time"] = []
    dict_scores["fit_time"] = []
    
    set_size = int( (1.0/K) * X_dataset.shape[0])
    
    for i in range(K):
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        start_time = time.time()
        X_train = np.block([[X_dataset[:l_bound]],[X_dataset[r_bound:]]])
        y_train = np.block([[y_dataset[:l_bound]],[y_dataset[r_bound:]]])
        X_val = X_dataset[l_bound:r_bound]
        y_val = y_dataset[l_bound:r_bound]
        
        estimator.fit(X_train, y_train.ravel())
        end_time = time.time()
        fit_time_elapsed = end_time - start_time
          
        start_time = time.time()
        y_pred = estimator.predict(X_val)
        score = scoring(y_val, y_pred)
        end_time = time.time()
        
        score_time_elapsed = end_time - start_time

        dict_scores["score"].append(score)
        dict_scores["score_time"].append(score_time_elapsed)
        dict_scores["fit_time"].append(fit_time_elapsed)
    
    #print(dict_scores)
    
    mean_score = statistics.mean(dict_scores["score"])
    
    return mean_score
