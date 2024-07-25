import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder


DATAPATH = r'C:\Users\lollo\Documents\Università\Thesis\Code\data\\corr\\'

def wrapper(args):
   
    num_colors = args['num_colors']
    grid_size = args['grid_size']
    betas = args['betas']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate'] 
    
    data = []
    
    m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=1)
    
    def sort_lengths_and_visits(lengths, visits):
        # Pair each length with its corresponding visits
        paired_lengths_visits = list(zip(lengths, visits))
        
        # Sort the list of tuples by length
        sorted_pairs = sorted(paired_lengths_visits, key=lambda x: x[0])
        
        # Separate the sorted lengths and the corresponding sorted visits
        sorted_lengths = [pair[0] for pair in sorted_pairs]
        sorted_visits = [pair[1] for pair in sorted_pairs]
        
        return sorted_lengths, sorted_visits

    def nu(max_K=100):  # manyally adjust!
        _, lengths, visits = m.loop_builder()
        # Sort lengths and apply the permutation to visits
        sorted_lengths, sorted_visits = sort_lengths_and_visits(lengths[0], visits[0])
        
        
        # Step 3: Calculate the sum of visits for each K from 2 to the max loop length
        #max_length = sorted_lengths[-1] if sorted_lengths else 0
        cumulative_visits = []
        #if max_K == None:
        #    max_K = max_length
        Ks = [ 2*np.ceil((max_K/2)**gamma) for gamma in np.linspace(1e-2, 1, 100)  ]
        for K in Ks:
            # Find all visits where the corresponding length >= K
            visits_ge_K = [visit for length, visit in zip(sorted_lengths, sorted_visits) if length >= K]
            
            # Sum the visits
            sum_visits = sum(visits_ge_K)
            
            # Store the result
            cumulative_visits.append(sum_visits)
        
        return np.array(cumulative_visits) /  m.grid_size**2
    
    observables = {
        'nu': nu
    }
    
    for beta in (t := tqdm(betas)): 
        t.set_description(f'{beta = }') 
       
        m.clear_data()
        m.beta = beta
        
        if beta == 1/1.2:
            # 200M
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
        else:
            # 100M
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
         
        # sample 
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)

        data.append(m.data['nu'])

    return np.transpose(data, (1, 0, 2)) # right shape for arviz  (betas, draws, K) -> (draws, betas, K) 

if __name__ == '__main__':
    
    params = {
        'num_colors': 2,
        'grid_size': 128, #[32, 64, 96, 128], #, 192, 256], 
        'betas': [1/1.2, 1/1.1, 1.0],
        'eq_steps': 25_000_000, #30_000_000,
        'run_steps': 25_000_000, #50_000_000,
        'sample_rate': 100_000
        }

    num_cores = cpu_count() // 2
    N = 1
    
    print(f"Running {N} trial(s) for each of the {num_cores} cores.")  
    
    start = time.time()
    with Pool(processes=num_cores) as pool:
        results = pool.map(wrapper, [params] * N * num_cores)
    end = time.time()
    
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Print the elapsed time in hh:mm:ss format
    print(f'Total elapsed time {hours:02d}:{minutes:02d}:{seconds:06.3f}')
    
    # build a dictionary 
    data = {
       'params': params,
       'results': results
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    #file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    file_name = f"XY_nu_lb_{params['grid_size']}_{timestamp}.json"
    print(f'Saving results to file {DATAPATH + file_name}')
    

    with open(DATAPATH + file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)

        