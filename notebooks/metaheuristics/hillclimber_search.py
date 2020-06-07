import time

def hillclimber_search(random_start,
                  random_neighbour,
                  scoring,
                  max_iterations, 
                  time_limit_s,
                  debug=True):
    
    if debug:
        print("Initiating...")
    
    best_state = random_start()
    best_score = scoring(best_state)
    
    initial_time = time.time()
    
    iterations = 0
    
    if debug:
        print("Starting search...")
    
    while max_iterations > iterations and (time.time() - initial_time) < time_limit_s:
        
        iterations += 1
        
        next_state = random_neighbour(best_state)
        
        score = scoring(next_state)
        
        if score > best_score:
            best_state = next_state
            best_score = score  
            
            if debug:
                print(f"{iterations}: Found better solution [k = {best_state}, score: {best_score}]")
        
        if debug and iterations % 10 == 0:
            print("Processing...")
    
    if debug:
        print(f"Best solution: state = {best_state}, score = {best_score}")
        
    return best_state