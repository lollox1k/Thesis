import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import trange, tqdm
import time 
import json 
from datetime import datetime

from RandomLoop import StateSpace, NumpyEncoder, check_connectivity


DATAPATH = r'C:\Users\lollo\Documents\Università\Thesis\Code\data\\'

def wrapper(args):
    
    data = []
    STEPS = args['steps']
    RATE = args['rate']
    for beta in args['betas']:
        m = StateSpace(num_colors=3, grid_size=64, beta=beta)

        #def loops():
        #    _, lengths = m.loop_builder()
        #    return max([ max(l) for l in lengths]), np.mean([l for sublist in lengths for l in sublist])
        
        def connectivity(d=8, grid_size=64):
            # get M random pairs at ditance d
            x_vertices = [tuple(np.random.randint(1, grid_size + 1, 2)) for _ in range(1000)]
            y_vertices = []
            for x in x_vertices:
                if x[0] + d >= grid_size:
                    y_vertices.append(  (x[0] - d, x[1]) )
                else:
                    y_vertices.append(  (x[0] + d, x[1]) )
            tot = 0
            for i in range(1000):
                tot += check_connectivity(m.grid, 0, x_vertices[i], y_vertices[i])
            return tot / 1000
        
        observables = {
            #'longest_loop': longest_loop
            #'mean_loop_length': m.mean_loop_length,
            #'loops': loops
        }
        
        # sample 
        m.step(num_steps=STEPS, sample_rate=RATE, observables=[connectivity], progress_bar=True)
        #m.step(num_steps=STEPS, sample_rate=RATE, observables=observables, progress_bar=True)
        #m.step(num_steps=STEPS, sample_rate=RATE, observables=observables, progress_bar=True)
        #m.step(num_steps=STEPS, sample_rate=RATE, observables=observables, progress_bar=True)
        
        data.append(m.data['connectivity'])
    
    return data # right shape for arviz  (betas, draws) -> (draws, betas)

if __name__ == '__main__':
    
    params = {
        'num_colors': 3,
        'grid_size': 64, 
        'betas': [2, 3, 4],
        'steps': 25_000_000,
        'rate': 250_000
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
        'data': results,
    }
    
   # Generate a timestamp string without colons
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    #file_name = f"run_{params['num_colors']}_{params['grid_size']}_{timestamp}.json"
    file_name = f"random_init_test_{timestamp}.json"
    print(f'Saving results to file {DATAPATH + file_name}')
    
    with open(DATAPATH + file_name, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)