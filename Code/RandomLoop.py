import json
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.auto import trange, tqdm

from numba import jit

from itertools import cycle
import logging
import time
from functools import wraps
from collections import deque

import colorsys
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.text import Text


"""
The code is implemented as a Python class called StateSpace, which contains methods for initializing the grid, running the simulation, and visualizing the results. 
The class also includes methods for saving and loading the state of the simulation to/from a file, and for calculating various statistics about the loops formed by each color.
"""


#################### global variables ########################

GAMMA = 1.5   # changes the gradient of the colormap, high GAMMA means similar colors, big gap with the background, GAMMA = 1 means the gradient is linear which cause low visibility sometimes
plt.style.use("dark_background")  # set dark background as default style
plt.rcParams['animation.embed_limit'] = 128 # Set the limit to 100 MB (default is 21 MB)

# How the grid id saved.
#
# m = (m_r, m_g, m_b) 3 colors
# the grid is from (0,0) to (n,n) make NxN square, each is a 2 dim list, each vertex has an horizontal and vertical edges like so:
#               0
#             ----(x,y)
#                   |    1
#                   |
#
# the 4 directions are encoded as:
#                     0
#                  -------
#             3    |     |   1
#                  |_____|
#                     2
#

#################### logger (disabled) ########################

'''
# set up logging to file
logging.basicConfig(
    filename='performance.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def log_time(func):
    """
    A decorator that logs the mean time a function takes to execute in milliseconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'total_time'):
            wrapper.total_time = 0  # Total time spent in all calls
            wrapper.calls = 0  # Number of calls

        start_time = time.perf_counter_ns()  # Capture start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.perf_counter_ns()  # Capture end time

        elapsed_time_ns = end_time - start_time   # Calculate elapsed time in ns
        wrapper.total_time += elapsed_time_ns  # Accumulate total time
        wrapper.calls += 1  # Increment call count
        mean_time_ns = wrapper.total_time / wrapper.calls  # Calculate mean time
        if wrapper.calls % 100_000 == 0:
            logging.info(f"{func.__name__} executed in {elapsed_time_ns:.2f} ns, mean execution time: {mean:.8f} ns over {wrapper.calls} call(s)")
        return result
    return wrapper
'''

#################### class ########################

class StateSpace:
    """
    A class representing the state space of a coloring problem on a grid.
    """
    def __init__(self, num_colors, grid_size, beta, init=0, bc=0, algo='metropolis'): 
        """
        Initialize the state space with the given parameters.

        Args:
          num_colors: The number of colors to be used in the simulation.
          grid_size: The size of the grid.
          beta: A parameter used in the acceptance probability calculation.
          init: The initialization method for the grid. It can be 'random' or a number.
          bc: The boundary condition for the grid. It can be 'random' or a number.
          algo: The algorithm to be used for the simulation. It can be 'metropolis' or 'glauber'.
        """
    
        self.grid_size = grid_size
        self.V = (grid_size-1)**2
        self.num_colors = num_colors
        self.beta = beta
        self.accepted = 0
        self.rejected = 0
        self.algo = algo
        self.sample_rate = None
        self.stop = False
        self.data = {}  
        self.shape = (num_colors, grid_size+2, grid_size+2, 2)  #the +2 is for free bc  we work only on vertices from  (1,grid_size+1)
        self.grid = 2*np.random.randint(0, 10, self.shape, dtype=int) if bc == 'random' else bc*np.ones(self.shape, dtype=int)
      
        # set the initial state
        if init == 'random':
            self.random_init()
        elif init == 'snake':
            self.build_snake()
        elif init == 'donut':
            self.build_donut()
        else:
            self.uniform_init(init)

    def random_init(self): # random with mean links approx beta
        self.grid[:, 1:-1, 1:-1, :] = 2*np.random.randint(0, 2 + self.beta // (2 * self.num_colors), size = (self.num_colors, self.grid_size, self.grid_size, 2), dtype=int)
        # bottom border
        self.grid[:, 1:-1, 0, 0] = 2*np.random.randint(0, 2 + self.beta // (2 * self.num_colors), size = (self.num_colors, self.grid_size))
        # left border
        self.grid[:, 0, 1:-1, 1] = 2*np.random.randint(0, 2 + self.beta // (2 * self.num_colors), size = (self.num_colors, self.grid_size))
        
        # apply random transformations
        for _ in range(self.V):
             # choose a random color 
            c = np.random.randint(0, self.num_colors)
            # choose a random square
            s = np.random.randint(0, self.grid_size, size=2)
            S = np.zeros(4, dtype=int)
            # handle boundary cases 
            if s[0] == 0:
                if s[1] == 0:
                    transformations = [(0,0,0,0)]
                S[1] = self.grid[c, s[0], s[1], 1]
                if S[1] >= 2:
                    transformations = [(0, 2, 0, 0), (0, -2, 0, 0)]
                else:
                    transformations = [(0, 2, 0, 0)]
                    
            elif s[1] == 0:
                S[0] = self.grid[c, s[0], s[1], 0]
                if S[0] >= 2:
                    transformations = [(2, 0, 0, 0), (-2, 0, 0, 0)]
                else:
                    transformations = [(2, 0, 0, 0)]
            else:
                # get links of color c on each side of square s
                S[0] = self.grid[c, s[0], s[1], 0]
                S[2] = self.grid[c, s[0], s[1]-1, 0]
                S[1] = self.grid[c, s[0], s[1], 1]
                S[3] = self.grid[c, s[0]-1, s[1], 1]

                # get list of all possible transformation
                transformations = get_possibile_transformations(S) 
            
            # pick uniformly a random transformation
            M = len(transformations)   # num of possible transformation of current state, compute only once! we also need it to compute tha ratio M/M_prime in acceptance_prob
            index = np.random.randint(0,M)
            X = transformations[index]
            self.square_transformation(c, s, X)
      

    def uniform_init(self, k):
        self.grid[:, 1:-1, 1:-1, :] = k*np.ones((self.num_colors, self.grid_size, self.grid_size, 2), dtype=int)
        
    def step(self, num_steps=1, sample_rate=10_000, beta=None, observables=None, progress_bar=True): 
        """
        Update the grid for a given number of steps.

        Args:
          num_steps: The number of steps to run the simulation for.
          progress_bar: Whether to show a progress bar during the simulation.
          sample_rate: The rate at which to sample observables during the simulation.
          observables: A list of functions or tuples of function and string that calculate observables to be measured during the simulation.
        """
        # update beta value
        self.beta = beta if beta is not None else self.beta
        
        # add some info to the data dictionary
        self.data['num_steps'] = self.data.get('num_steps', 0) + num_steps # if it's the first run, set the number of steps, otherwise add it.
        self.data['beta'] = self.beta
        self.data['grid_size'] = self.grid_size
        self.sample_rate = sample_rate
        
        # prepare the data dictionary to store samples
        obs_is_dict = isinstance(observables, dict) 
        if observables is not None: 
            for ob in observables:
                if callable(ob):
                    # Case where observables is a list of callable objects
                    self.data.setdefault(ob.__name__, [])
                elif isinstance(ob, str):
                # New case: observables includes method names as strings
                    self.data.setdefault(ob, [])
                else:
                    print(f'Observable {ob} is not valid.')
        else:
            observables = []
        
        # generate random numbers before looping to improve performance by x5 !!! WARNING uses lots of RAM (100M steps is 600Mb)
        squares = np.random.randint(0, self.grid_size+1, size = (num_steps, 2)) 
        colors = np.random.randint(0, self.num_colors, size = num_steps)
        indexs = [np.random.randint(0, M+1, size = num_steps) for M in range(0, 13)]
        
        for i in trange(num_steps, disable = not progress_bar):  
            # store data every sample_rate steps
            if i % sample_rate == 0:   
                for ob in observables:
                    if obs_is_dict:
                        # When observables is a dictionary
                        self.data[ob].append(observables[ob]())
                    elif callable(ob):
                        # When observables is a list of callable objects
                        self.data[ob.__name__].append(ob())
                    elif isinstance(ob, str):
                        # New case: When observables includes method names as strings
                        method = getattr(self, ob)
                        self.data[ob].append(method())
                if self.stop: # flag to stop the chain 
                    self.stop = False
                    return None 
                
            # choose a random color 
            c = colors[i]
         
            # choose a random square
            s = squares[i] 

            S = np.zeros(4, dtype=int)
            
            # handle boundary cases 
            if s[0] == 0:
                if s[1] == 0:
                    transformations = [(0,0,0,0)]
                S[1] = self.grid[c, s[0], s[1], 1]
                if S[1] >= 2:
                    transformations = [(0, 2, 0, 0), (0, -2, 0, 0)]
                else:
                    transformations = [(0, 2, 0, 0)]
                    
            elif s[1] == 0:
                S[0] = self.grid[c, s[0], s[1], 0]
                if S[0] >= 2:
                    transformations = [(2, 0, 0, 0), (-2, 0, 0, 0)]
                else:
                    transformations = [(2, 0, 0, 0)]
            else:
                # get links of color c on each side of square s
                S[0] = self.grid[c, s[0], s[1], 0]
                S[2] = self.grid[c, s[0], s[1]-1, 0]
                S[1] = self.grid[c, s[0], s[1], 1]
                S[3] = self.grid[c, s[0]-1, s[1], 1]

                # get list of all possible transformation
                transformations = get_possibile_transformations(S) 
            
            # pick uniformly a random transformation
            M = len(transformations)   # num of possible transformation of current state, compute only once! we also need it to compute tha ratio M/M_prime in acceptance_prob
            index = indexs[M-1][i] 
            X = transformations[index]
           
            if np.random.rand() <= acceptance_prob_optimized(S, M, s, X, c, self.beta, self.num_colors, self.grid):  
                self.accepted += 1
                self.square_transformation(c, s, X)
            else:
                self.rejected += 1
            

    def square_transformation(self, c, s, X):    # X = (a_1, a_2, a_3, a_4) like in the thesis
        """
        Apply a transformation X to a square s of color c.

        Args:
          c: The color of the square to be transformed.
          s: The square to be transformed.
          X: The transformation to be applied to the square.
        """
        #top
        self.grid[c, s[0], s[1], 0] += X[0]
        #right
        self.grid[c, s[0], s[1], 1] += X[1]
        #bottom
        self.grid[c, s[0], s[1]-1, 0] += X[2]
        #left
        self.grid[c, s[0]-1, s[1], 1] += X[3]
        
    def get_grid(self):
        """
        Return the current state of the grid.

        Returns:
          A copy of the current state of the grid.
        """
        return np.copy(self.grid)
    
    def get_local_time(self, x, y):
        """
        Calculate the local time for a given square at position (x, y).

        Args:
          x: The x-coordinate of the square.
          y: The y-coordinate of the square.

        Returns:
          The local time for the square at position (x, y).
        """
        local_time = 0
        for c in range(self.num_colors):
            local_time += self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0]
        return local_time // 2

    def get_local_time_i(self, c, x, y):
        """
        Calculate the local time for a given square at position (x, y) for color c.

        Args:
          c: The color for which to calculate the local time.
          x: The x-coordinate of the square.
          y: The y-coordinate of the square.

        Returns:
          The local time for the square at position (x, y) for color c.
        """
        return (self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0] ) // 2

    def max_links(self):
        """
        Return the maximum number of links for each color.

        Returns:
          The maximum number of links for each color.
        """
        return np.max(self.grid[:, 1:-1, 1:-1, :], axis=(1,2,3))
    
    def mean_links(self):
        """
        Return the mean number of links for each color.

        Returns:
          The mean number of links for each color.
        """
        return np.mean(self.grid[:, 1:-1, 1:-1, :], axis = (1,2,3))
    
    def std_links(self):
        """
        Return the mean number of links for each color.

        Returns:
          The mean number of links for each color.
        """
        return np.std(self.grid[:, 1:-1, 1:-1, :], axis = (1,2,3))

    def mean_local_time(self):
        """
        Return the mean local time for the grid.

        Returns:
          The mean local time for the grid.
        """
        total_local_time = 0
        for x in range(1,self.grid_size+1):
            for y in range(1,self.grid_size+1):
                total_local_time += self.get_local_time(x,y)
        return total_local_time / self.V
    
    def concentration_coeff(self):
        tot = 0
        for x in range(1,self.grid_size+1):
            for y in range(1,self.grid_size+1):
                lt = self.get_local_time(x, y)
                if lt != 0:
                    for c in range(self.num_colors):
                        for cc in range(c):
                            tot += abs( self.get_local_time_i(c, x, y) -  self.get_local_time_i(cc, x, y)) / lt
                    
        return tot / ((self.num_colors-1) * self.V)
    def loop_builder(self, v=None):
        """
        Build loops for each color in the grid.

        Args:
          v (Optional[Tuple[int, int]]): The starting vertex for the loop. Defaults to None.

        Returns:
          If v is None, return a tuple of two lists: the first list contains a list of loops for each color, where each loop is represented as a list of tuples of integers representing the (x, y) coordinates of the vertices in the loop; the second list contains the lengths of all loops.
          If v is a Tuple, return loops and lenghts of loops that visit v.
        """
        loops = []
        lenghts = []
        for c in range(self.num_colors):
            color_loops = []
            color_lengths = []
            #copy the grid 
            G = np.copy(self.grid[c])
            # set to zero in the boundary  
            G[0, :, 0] = 0  
            G[:, 0, 1] = 0

            # check if there are links
            if v is None:
                nz = G.nonzero()
                non_zero = len(nz[0])    
            else:
                non_zero = G[v[0], v[1], 0] + G[v[0], v[1], 1] + G[v[0], v[1]+1, 1] + G[v[0]+1, v[1], 0]
                
            if non_zero == 0:
                color_lengths.append(0)
                    
            while non_zero > 0:
                #pick first non-zero and unvisited edge
                if v is None:
                    nz = G.nonzero()
                    non_zero = len(nz[0])
                    if non_zero != 0:
                        x, y  = nz[0][0], nz[1][0]
                    else:
                        break
                else:
                    non_zero = G[v[0], v[1], 0] + G[v[0], v[1], 1] + G[v[0], v[1]+1, 1] + G[v[0]+1, v[1], 0]
                    if non_zero != 0:
                        x, y = v[0], v[1] 
                    else:
                        break
              
                starting_vertex = (x,y) 
                current_loop = []
                
                # first step outside loops | why?
                dir = []
                
                top = G[x,y+1,1]
                right = G[x+1,y,0]
                bottom = G[x,y,1]
                left = G[x,y,0]
                
                if top > 0:
                    dir.extend([0]*top)
                if right > 0:
                    dir.extend([1]*right)
                if bottom > 0:
                    dir.extend([2]*bottom)
                if left > 0:
                    dir.extend([3]*left)
                
                rand_dir = np.random.choice(dir)
                
                if rand_dir == 0:
                    # remove one link
                    G[x,y+1,1] -= 1
                    #move there
                    y += 1
                elif rand_dir == 1:
                    G[x+1,y,0] -= 1
                    x += 1
                elif rand_dir == 2:
                    G[x,y,1] -= 1
                    y -= 1
                elif rand_dir == 3:
                    G[x,y,0] -= 1
                    x -= 1
                    
                length = 0
                while True:
                    current_loop.append((x,y))
                    length += 1
                    # look if we can trav in each of the 4 directions: top = 0, right = 1, down = 2 and left = 3 with prob eq. to num_links/Z
                    dir = []
                    
                    top = G[x,y+1,1]
                    right = G[x+1,y,0]
                    bottom = G[x,y,1]
                    left = G[x,y,0]
                    
                   
                    if top > 0:
                        dir.extend([0]*top)
                    if right > 0:
                        dir.extend([1]*right)
                    if bottom > 0:
                        dir.extend([2]*bottom)
                    if left > 0:
                        dir.extend([3]*left)
                    
                    
                    if (x,y) == starting_vertex:
                        if np.random.rand() <= 1/(len(dir)+1):
                            color_lengths.append(length)
                            break
                    
                    rand_dir = np.random.choice(dir)
                    
                    if rand_dir == 0:
                        # remove one link
                        G[x,y+1,1] -= 1
                        #move there
                        y += 1
                    elif rand_dir == 1:
                        G[x+1,y,0] -= 1
                        x += 1
                    elif rand_dir == 2:
                        G[x,y,1] -= 1
                        y -= 1
                    elif rand_dir == 3:
                        G[x,y,0] -= 1
                        x -= 1
                     
                color_loops.append(current_loop)
            loops.append(color_loops)
            lenghts.append(color_lengths)
            
        return loops, lenghts
    
    def compute_corr(self):
        """
        Checks if vertices of increasing distance are connected by a loop.

        Returns: A list of 1 or 0 (1 if vertices are connected, 0 if not).
        """
        is_connected = []

        # generate a random point, then find one at distance d

        start_vertices = [tuple(np.random.randint(1, self.grid_size + 1, 2)) for _ in range(self.grid_size // 2 + 1)]
        end_vertices = []
        
        for i in range(self.grid_size // 2 + 1):
            if np.random.rand() <= 0.5: # vertical
                if start_vertices[i][1] + i <= self.grid_size:
                    end_vertices.append(  (start_vertices[i][0], start_vertices[i][1] + i) )
                else:
                    end_vertices.append(  (start_vertices[i][0], start_vertices[i][1] - i) )
            else: # horizontal
                if start_vertices[i][0] + i <= self.grid_size:
                    end_vertices.append(  (start_vertices[i][0] + i, start_vertices[i][1]) )
                else:
                    end_vertices.append(  (start_vertices[i][0] - i, start_vertices[i][1]) )
          
        for i in range(self.grid_size // 2 + 1):
            is_connected.append(check_connectivity(self.grid, self.num_colors, start_vertices[i], end_vertices[i]))

        return is_connected
    
    def mean_loop_length(self):
        """
        Return the mean loop length for the grid.

        Returns:
          The mean loop length for the grid.
        """
        _, lengths = self.loop_builder()
        return [ sum(l) / len(l) for l in lengths]
                
    def check_state(self):
        """
        Check if the current state of the grid is legal.

        Returns:
          True if the current state of the grid is legal, False otherwise.
        """
        for c in range(self.num_colors):
            for x in range(1,self.grid_size+1):
                for y in range(1,self.grid_size+1):
                    if (self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0]) % 2 != 0:
                        print('###  illegal state!  ###')
                        return False 
        return True 
    
    def find_connected_components(self, v=None):
        
        """
        Find connected components in each 2D grid of the 4D array.
        
        :param binary_array: 4D numpy array with shape (num_colors, grid_size, grid_size, 2)
        :return: List of lists containing the number of connected components for each color
        """
        binary_array = self.grid != 0 
        binary_array = binary_array.astype(int)
            
        return find_components_with_vertices(binary_array, v)
    
    def check_percolation(self, color=0):
        """
        Checks for percolation
        """
        # generate perc configuration
        perc_conf = np.where( self.grid[color] >= 1, 1, 0)
        
        for y in range(self.grid_size + 1):
            if bfs(perc_conf, (0, y)):
                return 1 
        return 0 
    
    def plot_one_color(self, c, cmap, ax, alpha=1.0, linewidth = 1.0):
        """
        Optimized function to plot grid lines of color c using batch drawing with LineCollection for efficiency.
        
        Args:
          c: The color to be plotted.
          cmap: The colormap to be used for plotting.
          ax: The axis on which to plot the grid.
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        # Initialize lists to collect line segments
        segments = []

        # Collect line segments for horizontal and vertical lines
        for x in range(len(self.grid[0][0])):
            for y in range(len(self.grid[0][0])):
                # horizontal lines
                if self.grid[c][x][y][0] != 0:
                    segments.append([(x - 1, y), (x, y)])
                # vertical lines
                if self.grid[c][x][y][1] != 0:
                    segments.append([(x, y), (x, y - 1)])

        # Assuming colors are normalized between 0 and 1, adjust as needed
        line_colors = [cmap(self.grid[c][x][y][z]) for x in range(len(self.grid[0][0])) for y in range(len(self.grid[0][0])) for z in range(2) if self.grid[c][x][y][z] != 0]

        # Create a LineCollection
        lc = LineCollection(segments, colors=line_colors, linewidths=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc
    
    def plot_loop(self, ax, c, loop, colors=None, alpha=0.25, linewidth=1.5): 
        """
        Highlights a loop(s) in a given color c.

        Args:
          c: The color for which to plot.
          loop: The loop(s) to be plotted
          color: The color to be used for plotting the loop. 
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        if loop == []:
            return None
        elif type(loop[0]) ==  tuple:
            loop = [loop] 
            
        if colors == None:
            colors_gen = cycle([plt.cm.Set3(i) for i in range(12)])
            colors = [next(colors_gen) for _ in range(len(loop))] 
        # Initialize lists to collect line segments
        segments = []
       
        for l in loop:
            for i in range(len(l)-1):
                #segments.append([l[i][0], l[i+1][0]], [l[i][1], l[i+1][1]])
                segments.append([l[i], l[i+1]])
               
            #draw last link
            #segments.append( [l[0][0], l[-1][0]], [l[0][1], l[-1][1]]) #, linewidth=linewidth, color=color, alpha=alpha, label='length = {}'.format(len(l)))
            if np.sum( abs(l[-1][0] - l[0][0]) + abs(l[-1][1] - l[0][1]) ) == 1:
                segments.append([l[-1], l[0]])  
        #line_colors = [cmap(self.grid[c][x][y][z]) for x in range(len(self.grid[0][0])) for y in range(len(self.grid[0][0])) for z in range(2) if self.grid[c][x][y][z] != 0]
       
        # Create a LineCollection
        lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc
    
        
    def plot_loop_overlap(self, loops, figsize=(12,12), alpha=1, linewidth=1.5, colors=None, file_name=None):
        """
        Highlights a loop(s) in the overlapped grid

        Args:
          loop: The loop(s) to be plotted.
          color: The color to be used for plotting the loop. Default is yellow
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        for c in range(self.num_colors):
            # Define a colormap
            num_segments = int(self.max_links()[c]+1)
            cmap = create_cmap(self.num_colors, c, num_segments)

            self.plot_one_color(c, cmap, ax, alpha)
            self.plot_loop(ax, c, loops[c], colors=colors, alpha=alpha, linewidth=linewidth)
            
            ax.set_title(r'grid size = {}     $\beta$ = {}        steps = {:.2e}'.format(self.grid_size, self.beta, self.accepted + self.rejected))
            ax.set_xlim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            ax.set_ylim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            
            ax.axis('off')
            
            #ax.axis('square')
            # grid
            #ax.set_xticks(np.arange(1,self.grid_size+1), minor = True)
            #ax.set_yticks(np.arange(1,self.grid_size+1), minor = True)
            #ax.grid(which='both')
            #ax.axis('off')
        #save it
        if file_name != None:
            plt.savefig(file_name)
        plt.show()
        
    
    def plot_grid(self, figsize=(10,8), linewidth=1.0, colorbar=True, axis=False, file_name=None):
        """
        Plot the grid for all colors.

        Args:
          figsize: The size of the figure to be plotted.
          linewidth: The width of the lines in the plot.
          colorbar: Whether to show a colorbar in the plot.
          file_name: The name of the file to save the plot to.
        """
        # scale figsize base on num_colors
        figsize = (figsize[0]*self.num_colors, figsize[1])
        fig, axes = plt.subplots(1, self.num_colors, figsize=figsize, gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        
        # Adjust the space between subplots
        for c in range(self.num_colors):
            # Define a colormap
            num_segments = int(self.max_links()[c]+1)            #color dependet!
            cmap = create_cmap(self.num_colors, c, num_segments)
            
            # Create a ScalarMappable for colorbar
            norm = Normalize(vmin=0, vmax=num_segments)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Empty array, as we'll not use actual data

            self.plot_one_color(c, cmap, axes[c], linewidth=linewidth)
            axes[c].set_xlim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            axes[c].set_ylim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            #axes[c].axis('square')
            if not axis:
                axes[c].axis('off')                                        
            
            # Add colorbar
            if colorbar:
                # Add colorbar
                cbar = plt.colorbar(sm, ax=axes[c])
                cbar.set_ticks(  0.5 + np.arange(0, num_segments,1))
                cbar.set_ticklabels(list(range(0, num_segments)))
                #cbar.set_label('Color Mapping')

        fig.suptitle(r'grid size = {}     $\beta$ = {}        steps = {:g}'.format(self.grid_size, self.beta, self.accepted + self.rejected))
        #save it
        if file_name != None:
            plt.savefig(file_name)
        plt.show()

    def plot_overlap(self, figsize = (12,12), normalized = False, file_name = None, alpha = 0.7, linewidth = 1.0):
        """
        Plot the overlap of all colors in the grid.

        Args:
          figsize: The size of the figure to be plotted.
          normalized: Whether to normalize the colors in the plot.
          file_name: The name of the file to save the plot to.
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        for c in range(self.num_colors):
            # Define a colormap
            num_segments = int(self.max_links()[c]+1) if not normalized else 2
            cmap = create_cmap(self.num_colors, c, num_segments)

            self.plot_one_color(c, cmap, ax, alpha, linewidth)
            ax.set_title(r'grid size = {}     $\beta$ = {}        steps = {:.2e}'.format(self.grid_size, self.beta, self.accepted + self.rejected))
            ax.set_xlim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            ax.set_ylim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            
            ax.axis('off')
            
            #ax.axis('square')
            # grid
            #ax.set_xticks(np.arange(1,self.grid_size+1), minor = True)
            #ax.set_yticks(np.arange(1,self.grid_size+1), minor = True)
            #ax.grid(which='both')
            #ax.axis('off')
        #save it
        if file_name != None:
            plt.savefig(file_name)
        plt.show()
        
    def animate(self, frames=None, interval=50, normalized=False, file_name=None, alpha = 1, linewidth=1):
         
         # first check if we have data
        assert 'get_grid' in self.data, 'to generate an animation first sample the grid state using the method "get_grid"'
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # compute mean links at the end for an approximation of max links, to show a coerent cmap across all the animation
        max_links = np.ceil( 2 + 3 * self.mean_links()).astype(int)
        if normalized: 
                num_segments = [2]*self.num_colors 
        else:
            num_segments = max_links
        
        if frames == None:
            frames = len(self.data['get_grid'])
            
        def get_frame(i):
            # Instead of clearing and redrawing, make all previous artists invisible
            for artist in ax.collections:
                artist.set_visible(False)
            ax.set_title('step {}'.format(i * self.sample_rate))
            artists = [] 
            
            for c in range(self.num_colors):
                cmap = create_cmap(self.num_colors, c, num_segments[c])  # Assuming this is defined
                segments = []
                # Collect line segments...
                # Collect line segments for horizontal and vertical lines
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        # horizontal lines
                        if self.data['get_grid'][i][c][x][y][0] != 0:
                            segments.append([(x - 1, y), (x, y)])
                        # vertical lines
                        if self.data['get_grid'][i][c][x][y][1] != 0:
                            segments.append([(x, y), (x, y - 1)])

                line_colors = [cmap(self.data['get_grid'][i][c][x][y][z]) for x in range(self.grid_size) for y in range(self.grid_size) for z in range(2) if self.data['get_grid'][i][c][x][y][z] != 0]
                lc = LineCollection(segments, colors=line_colors, linewidths=linewidth, alpha=alpha)
                ax.add_collection(lc)
                # Include this LineCollection in the list of artists to be updated
                artists.append(lc)
      
            ax.set_xlim(-0.05 * self.grid_size, self.grid_size * 1.05)
            ax.set_ylim(-0.05 * self.grid_size, self.grid_size * 1.05)
            ax.axis('off')

            return artists
         
        animation = FuncAnimation(fig, get_frame, frames=frames, interval=interval, repeat=False, blit=True)
       
       
        return animation if file_name is None else animation.save(file_name)
    
    def summary(self):
        """
        Print a summary of the current state of the grid.
        """
        print('mean number of links: {}'.format(self.mean_links()))
        print('max number of links: {}'.format(self.max_links() ))
        print('mean local time: {}'.format(self.mean_local_time()))
        _, lengths = self.loop_builder() #stats on color 0
        
        mean_loop_lenghts = [np.mean(l) for l in lengths]
        max_loop_lenghts = [np.max(l) for l in lengths]
        
        print('mean loop length: {}'.format(mean_loop_lenghts))
        print('max loop length: {}'.format(max_loop_lenghts))
        steps = self.accepted + self.rejected
        if steps == 0:
            print('steps = {:g}   acceptance ratio = {:.6f}'.format(steps, 0))
        else:
            print('steps = {:g}   acceptance ratio = {:.6f}'.format(steps, self.accepted / (steps)))
    
    def save_data(self, file_name):
        """
        Save the current state of the grid to a file.

        Args:
          file_name: The name of the file to save the data to.
        """

        #self.data['num_colors'] = self.num_colors
        #self.data['grid_size'] = self.grid_size
        #self.data['beta'] = self.beta 
        #self.data['algo'] = self.algo 
        #self.data['sample_rate'] = self.sample_rate
    
        # save other attributes
        for attr, value in vars(self).items():
            if attr != 'data':
                self.data[attr] = value 
        
        # transform every data element to list
        #self.data = {key: list(value) for key, value in self.data.items()}

        with open(file_name, 'w') as file:
            json.dump(self.data, file, cls=NumpyEncoder)
            
    def load_data(self, file_name):
        """
        Load the state of the grid from a file.

        Args:
          file_name: The name of the file to load the data from.
        """
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        # load state into the grid
        self.grid = np.array(self.data['grid'])
        
        for attr in vars(self):
            if attr != 'data' and attr != 'grid':
                setattr(self, attr, self.data[attr])
    
    def clear_data(self):
        """
        Clears the data dictionary
        """
        self.data.clear()
            
            
    def build_donut(self):
        #building the donut state
        for c in range(self.num_colors):
            for x in range(1, self.grid_size // 4+1):
                for y in range(1, self.grid_size+1):
                    self.grid[c, x, y, 0] = 1
                    self.grid[c, x, y, 1] = 1
            for x in range(self.grid_size+1 - self.grid_size // 4, self.grid_size+1):
                for y in range(1, self.grid_size+1):
                    self.grid[c, x, y, 0] = 1
                    self.grid[c, x, y, 1] = 1
        for c in range(self.num_colors):
            for y in range(1, self.grid_size // 4+1):
                for x in range(1, self.grid_size+1):
                    self.grid[c, x, y, 0] = 1
                    self.grid[c, x, y, 1] = 1
            for y in range(self.grid_size+1 - self.grid_size // 4, self.grid_size + 1):
                for x in range(1, self.grid_size+1):
                    self.grid[c, x, y, 0] = 1
                    self.grid[c, x, y, 1] = 1

        # add missing links on the outside border
        for c in range(self.num_colors):
            for y in range(1, self.grid_size+1):
                self.grid[c,0, y, 1] = 1
                self.grid[c, y, 0, 0] = 1 
                
        # add missing links on the inside border
        for c in range(self.num_colors):
            for x in range(self.grid_size // 4, 3 * self.grid_size // 4 + 1):
                self.grid[c,x,3 * self.grid_size // 4, 0] = 1
                self.grid[c, 3 * self.grid_size // 4, x, 1] = 1 
                
        #make it a legal state
        self.grid *= 2
    
    
    def build_snake(self):
        # build concentric squares
        offset = self.grid_size // 2+1
        for c in range(self.num_colors):
            for k in range(1,offset):
                self.grid[c, k:-k, k-1, 0] = 1
                self.grid[c, k-1, k:-k, 1] = 1
                
                self.grid[c, (offset-k):(k-offset), offset + k -1, 0] = 1
                self.grid[c, offset + k -1, (offset-k):(k-offset), 1] = 1
                
        # swap some links to join the squares
        for c in range(self.num_colors):
            for k in range(1,offset-1):
                self.grid[c, k+1 , self.grid_size-k+1, 0] = 0
                self.grid[c, k+1 , self.grid_size-k, 0] = 0
                
                self.grid[c, k , self.grid_size-k+1, 1] = 1
                self.grid[c, k+1 , self.grid_size-k+1, 1] = 1
        
            
# some utils functions for plotting    

def generate_rgb_colors(n):
    """
    Generates n RGB colors in the form of (r, g, b) with r, g, b values in [0, 1],
    maximizing the perceptual difference between the colors, ensuring they are bright and vivid.
    """
    colors = []
    
    for i in range(n):
        hue = i / n
        saturation = 1  # Max saturation for vivid colors
        lightness = 0.5  # Lightness set to 0.5 for bright colors
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    
    return colors

def create_cmap(num_colors, color, n_bins):  # create a color map to show how many links are in a edge   ############## Gamma and cmap norm to fine tune 
    
    # get current style
    current_style = infer_style()
    if current_style == 'dark_background':
        background_color = (0,0,0)
        gamma = 1 / GAMMA
    else:
        background_color = (1,1,1)
        gamma = GAMMA
    
    target_color = generate_rgb_colors(num_colors)[color]
    '''
    if color == 0:
        target_color = (1, 0, 0)  # Red
    elif color == 1:
        target_color = (0, 1, 0)  # Green
    elif color == 2:
        target_color = (0, 0, 1)  # Blue
    else:
        target_color = (0.8, 0, 1)  # Magenta-like for any color > 2
    '''
    # Generate interpolated colors with gamma correction
    colors = np.array([np.linspace(background_color[i], target_color[i], n_bins) for i in range(3)])
    colors = np.power(colors, gamma).T  # Apply gamma correction and transpose to get the correct shape  gamma = 0.5 to brighten the cmap
    
    # Create the colormap
    cmap = ListedColormap(colors, name = 'my_cmap')
    return cmap


def infer_style():
    for style_name, style_params in plt.style.library.items():
        if all(key in plt.rcParams and plt.rcParams[key] == val for key, val in style_params.items()):
            return style_name
    return 'Default'

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
        

def bfs(perc_conf, x):
    """
    Perform a BFS from the given vertex x, marking all visited vertices. If vertex y is found, return 1
    Vertex x should be on the left/bottom of the grid
    """
    grid_size = perc_conf.shape[0]
    visited = np.zeros(shape=(grid_size, grid_size))
    queue = deque()
    
    # Set current vertex as visited, add it to the queue
    visited[x[0], x[1]] = 1 
    queue.append(x) 
    
    if x[0] == 0: # x is on the left border
        horiz = True 
    elif x[1] == 0:
        horiz = False 
    else:
        print(x) 
        
    # Iterate over the queue
    while queue:
        # Dequeue 
        currentVertex = queue.popleft()
        
        # check here
        if horiz:
            if currentVertex[0] == grid_size - 2:
                #print(f'Reached vertex {currentVertex}')
                return 1 
        else:
            if currentVertex[1] == grid_size - 2:
                #print(f'Reached vertex {currentVertex}')
                return 1 
            
        # Get all neighbours of currentVertex        
        # If an adjacent has not been visited, then mark it visited and enqueue it
        if perc_conf[currentVertex[0], currentVertex[1], 0] == 1 and not visited[currentVertex[0]-1, currentVertex[1]]:
            # left neighbour
            visited[currentVertex[0]-1, currentVertex[1]] = 1
            queue.append((currentVertex[0]-1, currentVertex[1]))
            
        if perc_conf[currentVertex[0], currentVertex[1], 1] == 1 and not visited[currentVertex[0], currentVertex[1]-1]:
            # bottom neighbour
            visited[currentVertex[0], currentVertex[1]-1] = 1
            queue.append((currentVertex[0], currentVertex[1]-1))
            
        if perc_conf[currentVertex[0]+1, currentVertex[1], 0] == 1 and not visited[currentVertex[0]+1, currentVertex[1]]:
            # right neighbour
            visited[currentVertex[0]+1, currentVertex[1]] = 1
            queue.append((currentVertex[0]+1, currentVertex[1]))
            
        if perc_conf[currentVertex[0], currentVertex[1]+1, 1] == 1 and not visited[currentVertex[0], currentVertex[1]+1]:
            # top neighbour
            visited[currentVertex[0], currentVertex[1]+1] = 1
            queue.append((currentVertex[0], currentVertex[1]+1))
              
    return 0

def dfs_with_components(binary_array, x, y, color, visited, component):
    """
    Perform DFS from the given vertex, marking all reachable vertices within the same component
    and adding them to the component list.
    
    :param binary_array: 4D numpy array with shape (num_colors, grid_size, grid_size, 2)
    :param x: Current x coordinate in the grid
    :param y: Current y coordinate in the grid
    :param color: Current color layer being processed
    :param visited: 2D array marking visited vertices for the current color layer
    :param component: List to store vertices belonging to the current component
    """
    grid_size = binary_array.shape[1]
    if x <= 0 or x >= grid_size or y <= 0 or y >= grid_size or visited[x, y]:
        return
    
    # Mark the current vertex as visited and add it to the component
    visited[x, y] = True
    component.append((x, y))
    
    # Horizontal edge to the right
    if y + 1 < grid_size and binary_array[color, x, y, 0] == 1:
        dfs_with_components(binary_array, x, y + 1, color, visited, component)
    
    # Vertical edge downward
    if x + 1 < grid_size and binary_array[color, x, y, 1] == 1:
        dfs_with_components(binary_array, x + 1, y, color, visited, component)
    
    # Note: This approach assumes directed edges. If edges are bidirectional,
    # you'd need to consider additional cases for left and upward edges.

def find_components_with_vertices(binary_array, v=None):
    """
    Find connected components for each color in binary_array and return the vertices
    in each connected component. If v is not None, returns just the connected components that contain v
    
    :param binary_array: 4D numpy array with shape (num_colors, grid_size, grid_size, 2)
    :return: List of lists containing the vertices of connected components for each color
    """
    num_colors, grid_size, _, _ = binary_array.shape
    components_per_color = []
    
    for color in range(num_colors):
        visited = np.zeros((grid_size, grid_size), dtype=bool)
        color_components = []
        if v is None:
            for x in range(grid_size):
                for y in range(grid_size):
                    if not visited[x, y]:
                        component = []
                        dfs_with_components(binary_array, x, y, color, visited, component)
                        color_components.append(component)
        else:
            component = []
            dfs_with_components(binary_array, v[0], v[1], color, visited, component)
            color_components.append(component)
            
        components_per_color.append(color_components)
        
    return components_per_color
           
#########################################################################################################
#
#     To improve performance, we redefine computationally heavy functions outside and use numba 
#
#########################################################################################################

@jit(fastmath=True, nopython=True, cache=True)
def get_possibile_transformations(S):
    # Initialize the list with transformations that are always valid
    transformations = [
        (1, 1, 1, 1),  # U+
        (2, 0, 0, 0),  # T+
        (0, 2, 0, 0)   # R+
    ]
    
    # Conditional transformations based on the state of S
    if S[0] >= 2:
        transformations.append((-2, 0, 0, 0))  # single top
    if S[1] >= 2:
        transformations.append((0, -2, 0, 0))  # single right
    if S[0] > 0 and S[2] > 0:
        transformations.append((-1, 1, -1, 1))  # swap top-bottom
    if S[1] > 0 and S[3] > 0:
        transformations.append((1, -1, 1, -1))  # swap right-left
    if S[2] > 0 and S[3] > 0:
        transformations.append((1, 1, -1, -1))  # increase top-right, decrease bottom-left
    if S[0] > 0 and S[1] > 0:
        transformations.append((-1, -1, 1, 1))  # increase bottom-left, decrease top-right
    if S[0] > 0 and S[3] > 0:
        transformations.append((-1, 1, 1, -1))  # rotate clockwise
    if S[1] > 0 and S[2] > 0:
        transformations.append((1, -1, -1, 1))  # rotate counter-clockwise
    if np.all(S > 0):
        transformations.append((-1, -1, -1, -1))  # uniform decrease
    
    return transformations


@jit(fastmath=True, nopython=True, cache=True)
def minimal_transformations(S): # x2 faster
    """
    Return a list of just minimal transformations for a given square S.

    Args:
        S: The square for which to generate all possible transformations.

    Returns:
        A list of all possible transformations for the given square.
    """
    
    transformations = [(1,1,1,1)] #, (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2)]
    
    if S[0] >= 2:
        transformations.append((-2,0,0,0))
    if S[1] >= 2:
        transformations.append((0,-2,0,0))
    
    #transformations = [(-1,-1,-1,-1), (2,0,0,0), (0,2,0,0)] if np.all( S >= 1) else [(2,0,0,0), (0,2,0,0)]
    
    return transformations 


@jit(fastmath=True, nopython=True, cache=True)
def get_local_time(grid, num_colors, x, y):
        """
        Calculate the local time for a given square at position (x, y).

        Args:
          x: The x-coordinate of the square.
          y: The y-coordinate of the square.

        Returns:
          The local time for the square at position (x, y).
        """
        local_time = 0
        for c in range(num_colors):
            local_time += grid[c, x, y, 0] + grid[c, x, y, 1] + grid[c, x, y + 1, 1] + grid[c, x + 1, y, 0]
        return local_time // 2

@jit(fastmath=True, nopython=True, cache=True)
def get_local_time_i(grid, c, x, y):
    """
    Calculate the local time for a given square at position (x, y) for color c.

    Args:
        c: The color for which to calculate the local time.
        x: The x-coordinate of the square.
        y: The y-coordinate of the square.

    Returns:
        The local time for the square at position (x, y) for color c.
    """
    return (grid[c, x, y, 0] + grid[c, x, y, 1] + grid[c, x, y + 1, 1] + grid[c, x + 1, y, 0] ) // 2

                      
@jit(fastmath=True, nopython=True, cache=True)
def acceptance_prob_optimized(S, M, s, X, c, beta, num_colors, grid):
    """
        Calculate the acceptance probability for a transformation X on a square s of color c.

        Args:
          S: The current state of the square s.
          M: The number of possible transformations for the current state.
          s: The square to be transformed.
          X: The transformation to be applied to the square.
          c: The color of the square to be transformed.
          beta: the parameter beta of the simulation.
          num_colors: number of colors in the grid
          grid: the grid state.

        Returns:
          The acceptance probability for the transformation X on the square s of color c.
        """
    S_p = S + np.array(X)
    M_prime = len(get_possibile_transformations(S_p)) 
        
    A = 0
    num_colors_half = num_colors / 2

    if np.array_equal(X, [1, 1, 1, 1]): # U+
        A = beta**4 / (16 * S_p[0]*S_p[1]*S_p[2]*S_p[3] * \
            (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1)) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1))) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1)
    elif np.array_equal(X, [-1, -1, -1, -1]): # U-
        A = (16 / beta**4) * S[0]*S[1]*S[2]*S[3] * \
            (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) - 1))
            
    elif np.array_equal(X, [2, 0, 0, 0]): # T+
        A = beta**2 / (4 * S_p[0] * (S[0]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]))) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1)
       
    elif np.array_equal(X, [-2, 0, 0, 0]): # T-
        A = 4 * S[0] * (S[0]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) - 1))
       
    elif np.array_equal(X, [0, 2, 0, 0]): # R+
        A = beta**2 / (4 * S_p[1] * (S[1]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1))) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1)
        
    elif np.array_equal(X, [0, -2, 0, 0]): # R-
        A = 4 * S[1] * (S[1]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) - 1))
            
    elif np.array_equal(X, [-1, 1, -1, 1]):
        A = S[0]*S[2] / (S_p[1]*S_p[3])
    elif np.array_equal(X, [1, -1, 1, -1]):
        A = S[1]*S[3] / (S_p[0]*S_p[2])
    elif np.array_equal(X, [-1, -1, 1, 1]):
        A = S[0]*S[1] / (S_p[2]*S_p[3]) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) / (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1)) * \
            (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) + 1) / (2*get_local_time_i(grid, c, s[0], s[1]) - 1)
    elif np.array_equal(X, [1, 1, -1, -1]):
        A = S[2]*S[3] / (S_p[0]*S_p[1]) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1) - 1) / (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) / (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) - 1)
    elif np.array_equal(X, [-1, 1, 1, -1]):
        A = S[0]*S[3] / (S_p[1]*S_p[2]) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) / (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1)) * \
            (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1) / (2*get_local_time_i(grid, c, s[0]-1, s[1]) - 1)
    elif np.array_equal(X, [1, -1, -1, 1]):
        A = S[2]*S[1] / (S_p[3]*S_p[0]) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) / (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1])) * \
            (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1) / (2*get_local_time_i(grid, c, s[0], s[1]-1) - 1)

    # Calculate the acceptance probability based on the algorithm type
    return M/M_prime * A # if algo == 'metropolis' else 1/(1 + M_prime/(M*A)) # min() is useless since we are doing accept/rejection sampling...


##### no interaction 

@jit(fastmath=True, nopython=True, cache=True)
def acceptance_prob_no_int(S, M, s, X, c, beta, num_colors, algo, grid):
    """
    Calculate the acceptance probability for a transformation X on a square s of color c.

    Args:
        S: The current state of the square s.
        M: The number of possible transformations for the current state.
        s: The square to be transformed.
        X: The transformation to be applied to the square.
        c: The color of the square to be transformed.
        beta: the parameter beta of the simulation.
        num_colors: number of colors in the grid
        grid: the grid state.

    Returns:
        The acceptance probability for the transformation X on the square s of color c.
    """
    S_p = S + np.array(X)
    M_prime = len(get_possibile_transformations(S_p))
    A = 0

    if np.array_equal(X, [1, 1, 1, 1]):
        A = beta**4 / (S_p[0]*S_p[1]*S_p[2]*S_p[3]) 
    elif np.array_equal(X, [-1, -1, -1, -1]):
        A = (1 / beta**4) * S[0]*S[1]*S[2]*S[3] 
    elif np.array_equal(X, [2, 0, 0, 0]):
        A = beta**2 / ( S_p[0] * (S[0]+1) )
    elif np.array_equal(X, [-2, 0, 0, 0]):
        A = S[0] * (S[0]-1) / (beta**2) 
    elif np.array_equal(X, [0, 2, 0, 0]):
        A = beta**2 / (S_p[1] * (S[1]+1) )
    elif np.array_equal(X, [0, -2, 0, 0]):
        A = S[1] * (S[1]-1) / (beta**2) 
    elif np.array_equal(X, [0, 0, 2, 0]):
        A = beta**2 / (S_p[2] * (S[2]+1) )
    elif np.array_equal(X, [0, 0, -2, 0]):
        A = S[2] * (S[2]-1) / (beta**2) 
    elif np.array_equal(X, [0, 0, 0, 2]):
        A = beta**2 / (S_p[3] * (S[3]+1))
    elif np.array_equal(X, [0, 0, 0, -2]):
        A = S[3] * (S[3]-1) / (beta**2) 
    elif np.array_equal(X, [-1, 1, -1, 1]):
        A = S[0]*S[2] / (S_p[1]*S_p[3])
    elif np.array_equal(X, [1, -1, 1, -1]):
        A = S[1]*S[3] / (S_p[0]*S_p[2])
    elif np.array_equal(X, [-1, -1, 1, 1]):
        A = S[0]*S[1] / (S_p[2]*S_p[3]) 
    elif np.array_equal(X, [1, 1, -1, -1]):
        A = S[2]*S[3] / (S_p[0]*S_p[1]) 
    elif np.array_equal(X, [-1, 1, 1, -1]):
        A = S[0]*S[3] / (S_p[1]*S_p[2]) 
    elif np.array_equal(X, [1, -1, -1, 1]):
        A = S[2]*S[1] / (S_p[3]*S_p[0])

    # Calculate the acceptance probability based on the algorithm type
    return min(1, M/M_prime * A) if algo == 'metropolis' else 1/(1 + M_prime/(M*A))


#####
@jit(fastmath=True, nopython=True, cache=True)
def check_connectivity(grid, num_colors, v1 = None, v2 = None):
        """
        Build loops for each color in the grid.

        If v1 and v2 are both None, return a list of loops for each color and a list of integers representing the lengths of all loops.
        If v1 and v2 are both not None, return 1 if there exists a loop that joins v1 and v2, and 0 otherwise.

        Args:
          v1 (Optional[Tuple[int, int]]): The starting vertex for the loop. Defaults to None.
          v2 (Optional[Tuple[int, int]]): The ending vertex for the loop. Defaults to None.

        Returns:
          If v1 and v2 are both None, return a tuple of two lists: the first list contains a list of loops for each color, where each loop is represented as a list of tuples of integers representing the (x, y) coordinates of the vertices in the loop; the second list contains the lengths of all loops.
          If v1 and v2 are both not None, return an integer indicating whether there exists a loop that joins v1 and v2 (1 if such a loop exists, 0 otherwise).
        """

        for c in range(num_colors):
            #copy the grid 
            G = np.copy(grid[c])
            nz = G.nonzero()
            non_zero = len(nz[0])
            
            if non_zero == 0:
                return 0
            
            while non_zero > 0:
                #pick first non-zero and unvisited edge
                nz = G.nonzero()
                non_zero = len(nz[0])
                if non_zero == 0:
                    break
                # choose a random vertex with non-zero incident links, or choose v1
                if v1 is None:
                    rand_index = np.random.randint(0, non_zero)
                    x, y  = nz[0][rand_index], nz[1][rand_index]
                else:
                    x, y = v1[0], v1[1]
                    if (x,y) == v2:
                        return 1
                    
                starting_vertex = (x,y)
                
                # first step outside loops
                dir = []
                
                top = G[x,y+1,1]
                right = G[x+1,y,0]
                bottom = G[x,y,1]
                left = G[x,y,0]
                
                if top > 0:
                    dir.extend([0]*top)
                if right > 0:
                    dir.extend([1]*right)
                if bottom > 0:
                    dir.extend([2]*bottom)
                if left > 0:
                    dir.extend([3]*left)
                
                if len(dir) == 0:
                    return 0 
                
                # pick a random dir with prob prop to num_links  
                rand_dir = np.random.choice(np.array(dir))
                
                if rand_dir == 0:
                    # remove one link
                    G[x,y+1,1] -= 1
                    #move there
                    y += 1
                elif rand_dir == 1:
                    G[x+1,y,0] -= 1
                    x += 1
                elif rand_dir == 2:
                    G[x,y,1] -= 1
                    y -= 1
                elif rand_dir == 3:
                    G[x,y,0] -= 1
                    x -= 1
                    
                length = 0
                while True:
                    if (x,y) == v2:
                        return 1 
                    length += 1
                    # look if we can trav in each of the 4 directions: top = 0, right = 1, down = 2 and left = 3 with prob eq. to num_links/Z
                    dir = []
                    
                    top = G[x,y+1,1]
                    right = G[x+1,y,0]
                    bottom = G[x,y,1]
                    left = G[x,y,0]
                    
                    if top > 0:
                        dir.extend([0]*top)
                    if right > 0:
                        dir.extend([1]*right)
                    if bottom > 0:
                        dir.extend([2]*bottom)
                    if left > 0:
                        dir.extend([3]*left)

                    if (x,y) == starting_vertex:
                        if random.random() <= 1/(len(dir)+1):
                            break
                    
                    # pick a random dir with prob prop to num_links  
                    rand_dir = np.random.choice(np.array(dir))
                    
                    if rand_dir == 0:
                        # remove one link
                        G[x,y+1,1] -= 1
                        #move there
                        y += 1
                    elif rand_dir == 1:
                        G[x+1,y,0] -= 1
                        x += 1
                    elif rand_dir == 2:
                        G[x,y,1] -= 1
                        y -= 1
                    elif rand_dir == 3:
                        G[x,y,0] -= 1
                        x -= 1
                     
###




@jit(fastmath=True, nopython=True, cache=True)
def find_connected_components(matrix):
    def dfs(i, j, island):
        if i < 0 or i >= matrix.shape[0] or j < 0 or j >= matrix.shape[1] or matrix[i, j] == 0:
            return
        matrix[i, j] = 0  # Mark as visited
        island.append((i, j))
        
        dfs(i+1, j, island)  # Down
        dfs(i-1, j, island)  # Up
        dfs(i, j+1, island)  # Right
        dfs(i, j-1, island)  # Left

    components = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                current_island = []
                dfs(i, j, current_island)
                components.append(current_island)
    return components
   

# since running @jit functions for the first time is slow, we do a step of the chain at import
m = StateSpace(1, 10, 1)
m.step(progress_bar=False)
del m