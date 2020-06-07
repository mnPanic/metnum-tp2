import time

def random_search(random_start, 
                  random_state, 
                  scoring, 
                  max_iterations, 
                  time_limit_s):
     
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
        
        next_state = random_state()
        
        score = scoring(next_state)
        
        if score > best_score:
            best_state = next_state
            best_score = score  
            
            if debug:
                print(f"{iterations}: Found better solution [k = {best_state}, score: {best_score}]")
        
        if debug and iterations % 10 == 0:
            print("Processing...")
    
    return best_state