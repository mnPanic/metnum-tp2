import time
import numpy.random as rn

from tqdm import tqdm
from typing import List

def annealing(
        random_start,
        score_function,
        random_neighbour,
        acceptance_probability,
        temperature,
        max_state_reset_steps = 10,
        max_steps=1000,
        debug: bool = False
    ) -> (float, List[dict]):
    """
    Optimiza con simmulated annealing. 
    Devuelve dict con los states y scores.
    """

    best_state = random_start()
    best_score = score_function(best_state)
    state = best_state
    score = best_score

    history = []
    history.append({"step": 0, "state": state, "score": score, "best": False})
    
    if debug: print(f"initial: state = {state}, score = {score}")
    
    state_reset_steps = 0
    
    for step in tqdm(range(1, max_steps)):
        fraction = step / float(max_steps)
        T = temperature(fraction)
        new_state = random_neighbour(state, fraction)
        new_score = score_function(new_state)
        
        #if debug: print(f"#{step}: state = {new_state}, score = {new_score}")
        
        history.append({"step": step, "state": state, "score": score, "best": False})

        if acceptance_probability(score, new_score, T) > rn.random(): 
            # keep exploring from this state even if its not better 
            # than the best found so far
            state, score = new_state, new_score
            if score > best_score: 
                # save solution if it is better than the best found so far
                best_state, best_score = state, score
                
                if debug:
                    print(f"#{step} Found better solution [k = {state}, score = {score}]")

                # mark the last as best
                history[len(history) - 1]["best"] = True

                state_reset_steps = 0
            
        state_reset_steps += 1

        # reset state to best found so far after X iterations with no improvement
        if state_reset_steps == max_state_reset_steps:
            if debug:
                print("State reset")
            state = best_state
            state_reset_steps = 0
            
    if debug:
        print(f"Best solution: state = {best_state}, score = {best_score}")
     
    return best_state, history
