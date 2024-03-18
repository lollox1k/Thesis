import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder


def wrapper(args):
   
    trials = args['trials']
    num_colors = args['num_colors']
    grid_size = args['grid_size']
    beta = args['beta']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate']
    
    m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=beta)
    
    observables = [m.mean_links]

    for _ in range(trials):

        # restart the chain
        m.random_init()

        # reach equilibrium
        m.step(num_steps=eq_steps, progress_bar=True)

        # sample 
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=True)
   
    return np.sum(m.data['mean_links'], axis=1)

if __name__ == '__main__':

    params = {
        'trials': 1,
        'num_colors': 3,
        'grid_size': 64,
        'beta': 2,
        'eq_steps': 1_000_000,
        'run_steps': 10_000_000,
        'sample_rate': 100
        }

    N = cpu_count() // 2 # 16 threads but only 8 cpus

    print(f"Running { params['trials'] } trial(s) for each of the {N} cores.")  
    
    start = time.time()
    
    with Pool(processes=N) as pool:
        results = pool.map(wrapper, [params] * N)
    end = time.time()
    
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Print the elapsed time in hh:mm:ss format
    print(f'Total elapsed time {hours:02d}:{minutes:02d}:{seconds:06.3f}')
    
    # build a dictionary 
    
    data = {
       # 'params': params,
       # 'time': f'Total elapsed time {hours:02d}:{minutes:02d}:{seconds:06.3f}',
        'mean links': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
    file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    
    print(f'Saving results to file {file_name}')
    
    with open(file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)
    #with open('./data/corr_2_128_800.npy', 'wb') as f:
    #    
    #    np.save(f, np.mean(results, axis=0))
    
    