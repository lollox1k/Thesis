import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder

def wrapper(args):
   
    num_colors = args['num_colors']
    grid_size = args['grid_size']
    betas = args['betas']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate']
    
    local_times = []
    
    m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=1)
    
    for beta in (t := tqdm(betas)): 
        t.set_description(f'{beta = }') 
        
        m.clear_data()
        m.beta = beta
    
        # reach equilibrium 75M
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        
        # sample 
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=[m.local_time_corr], progress_bar=False)
        
        # save 
        local_times.append(np.transpose(np.array(m.data['local_time_corr']), axes = (1, 2, 0)))
        
    return local_times

if __name__ == '__main__':

    params = {
        'num_colors': 3,
        'grid_size': 64,
        'betas': [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6], 
        'eq_steps': 25_000_000,
        'run_steps': 25_000_000,
        'sample_rate': 100
        }

    N = cpu_count() // 2 # 16 threads but only 8 cpus

    print(f"Running {N} chains...")  
    
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
        'params': params,
        'time': f'Total elapsed time {hours:02d}:{minutes:02d}:{seconds:06.3f}',
        'data': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
    file_name = f"data/local_times/lt_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    
    print(f'Saving results to file {file_name}')
    
    with open(file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)

    
    