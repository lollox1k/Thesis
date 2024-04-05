import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder


DATAPATH = r'C:\Users\lollo\Documents\Università\Thesis\Code\data\\'

def wrapper(args):
   
    num_colors = args['num_colors']
    grid_sizes = args['grid_sizes']
    beta = args['beta']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate'] 
    
    data = []
   
    for grid_size in (t := tqdm(grid_sizes)): 
        t.set_description(f'{grid_size = }') 
        
        m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=beta, init=0)
        
        observables = {
        'origin_loop_length': lambda: np.sum(np.concatenate(m.loop_builder((grid_size//2, grid_size//2))[1]))  # sum   length of all loops touching the origin
        }

        # reach equilibrium 100M 
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        m.step(num_steps=eq_steps, progress_bar=False)
        
        if grid_size >= 128:
            # +100M
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
            m.step(num_steps=eq_steps, progress_bar=False)
        '''
        if grid_size >= 192:
            # +300M
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
        '''
        # sample 200M
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        '''
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=observables, progress_bar=False)
        '''
        data.append(m.data['origin_loop_length']) 
    
    return np.transpose(data, (1, 0)) # right shape for arviz  (g, draws) -> (draws, g)

if __name__ == '__main__':
    
    params = {
        'num_colors': 3,
        'grid_sizes': [32, 64, 96, 128], #, 192, 256], 
        'beta': 1.875,
        'eq_steps': 20_000_000,
        'run_steps': 20_000_000,
        'sample_rate': 1_000
        }

    num_cores = cpu_count() // 2
    N = num_cores
    
    print(f"Running {N} trial(s) for each of the {num_cores} cores.")  
    
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
        'origin_loop_length': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    #file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    file_name = f"origin_loops_{params['beta']}_{timestamp}.json"
    print(f'Saving results to file {DATAPATH + file_name}')
    
    with open(DATAPATH + file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)