import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder


DATAPATH = r'C:\Users\lollo\Documents\Università\Thesis\Code\data\\percolation\\'

def wrapper(args):
   
    num_colors = args['num_colors']
    grid_size = args['grid_size']
    betas = args['betas']
    eq_steps = args['eq_steps']
    run_steps = args['run_steps']
    sample_rate = args['sample_rate'] 
    
    data = []
   
    for beta in (t := tqdm(betas)): 
        t.set_description(f'{beta = }') 
        
        m = StateSpace(num_colors=num_colors, grid_size=grid_size, beta=beta, init=0)
        

        # reach equilibrium 10M 
        m.step(num_steps=eq_steps, progress_bar=False)
    
        # sample 10M
        m.step(num_steps=run_steps, sample_rate=sample_rate, observables=[m.check_percolation], progress_bar=False)
       
        data.append(m.data['check_percolation'])
    
    return np.transpose(data, (1, 0)) # right shape for arviz  (betas, draws) -> (draws, betas)

if __name__ == '__main__':
    
    params = {
        'num_colors': 3,
        'grid_size': 128, #[32, 64, 96, 128], #, 192, 256], 
        'betas': np.arange(1, 7, 0.5),
        'eq_steps': 10_000_000,
        'run_steps': 1_000_000,
        'sample_rate': 1_000
        }

    num_cores = cpu_count() // 2
    N = 4
    
    print(f"Running {N} trial(s) for each of the {num_cores} cores.")  
    
    start = time.time()
    with Pool(processes=N) as pool:
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
        'percolation': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    #file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    file_name = f"{params['grid_size']}_{timestamp}.json"
    print(f'Saving results to file {DATAPATH + file_name}')
    
    with open(DATAPATH + file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)