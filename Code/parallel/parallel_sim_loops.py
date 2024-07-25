import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder


DATAPATH = r'C:\Users\lollo\Documents\Università\Thesis\Code\data\\loops\\'

def wrapper(args):
   
    num_colors = args['num_colors']
    grid_size = args['grid_size']
    betas = args['betas']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate'] 
    
    data = []
    
    m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=1)
        
    for beta in (t := tqdm(betas)): 
        t.set_description(f'{beta = }') 
        
        m.clear_data()
        m.beta = beta
        
        if beta == 0.8:
            # reach equilibrium 
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            
        else:
            # reach equilibrium 
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
        
        def loops():
            _, lengths, visits = m.loop_builder()
            lengths_merged = [l for sublist in lengths for l in sublist]
            visits_merged = [l for sublist in visits for l in sublist]
            return np.mean(lengths_merged), np.max(lengths_merged), np.mean(visits_merged), np.max(visits_merged)

        observables = {
            'loops': loops
        }
        
        # sample 
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
       
        data.append(m.data['loops'])
    
    return np.transpose(data, (1, 0, 2)) # right shape for arviz  (betas, draws) -> (draws, betas)   #(betas, draws, 2) -> 

if __name__ == '__main__':
    
    params = {
        'num_colors': 2,
        'grid_size': 64, #[32, 64, 96, 128], #, 192, 256], 
        'betas': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        'eq_steps': 25_000_000,
        'run_steps': 10_000_000,
        'sample_rate': 10_000
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
       'results': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    #file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    file_name = f"XY_loops_{params['grid_size']}_{timestamp}.json"
    print(f'Saving results to file {DATAPATH + file_name}')
    
    with open(DATAPATH + file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)