import json
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.auto import trange, tqdm
import time
import logging
from functools import wraps
from numba import jit

import colorsys
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable


"""
The code is implemented as a Python class called StateSpace, which contains methods for initializing the grid, running the simulation, and visualizing the results. 
The class also includes methods for saving and loading the state of the simulation to/from a file, and for calculating various statistics about the loops formed by each color.
"""


#################### global variables ########################

GAMMA = 1.5   # changes the gradient of the colormap, high GAMMA means similar colors, big gap with the background, GAMMA = 1 means the gradient is linear which cause low visibility sometimes


# huge python class with everything inside
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

#################### class ########################

# set up logging to file
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def log_time(func):
    """
    A decorator that logs the average time a function takes to execute in milliseconds.
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
        average_time_ns = wrapper.total_time / wrapper.calls  # Calculate average time
        if wrapper.calls % 100_000 == 0:
            logging.info(f"{func.__name__} executed in {elapsed_time_ns:.2f} ns, average execution time: {average_time_ns:.8f} ns over {wrapper.calls} call(s)")
        return result
    return wrapper



class stateSpace:
    """
    A class representing the state space of a coloring problem on a grid.
    """
    def __init__(self, num_colors, grid_size, beta, init = 0, bc = 0, algo = 'metropolis'):  # 'glauber'      
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
        self.V = grid_size**2
        self.num_colors = num_colors
        self.beta = beta
        self.accepted = 0
        self.rejected = 0
        self.algo = algo
        self.sample_rate = 10_000
        
        self.data = {}  # dictionary to store data
        
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

    def random_init(self):
        self.grid[:, 1:-1, 1:-1, :] = 2*np.random.randint(0, 10, size = (self.num_colors, self.grid_size, self.grid_size, 2), dtype=int)
        # bottom border
        self.grid[:, 1:-1, 0, 0] = 2*np.random.randint(0, 10, size = (self.num_colors, self.grid_size))
        # left border
        self.grid[:, 0, 1:-1, 1] = 2*np.random.randint(0, 10, size = (self.num_colors, self.grid_size))

    def uniform_init(self, k):
        self.grid[:, 1:-1, 1:-1, :] = k*np.ones((self.num_colors, self.grid_size, self.grid_size, 2), dtype=int)
        
    def step(self, num_steps = 1, progress_bar = True, sample_rate = 10_000, observables = None): 
        """
        Update the grid for a given number of steps.

        Args:
          num_steps: The number of steps to run the simulation for.
          progress_bar: Whether to show a progress bar during the simulation.
          sample_rate: The rate at which to sample observables during the simulation.
          observables: A list of functions that calculate observables to be measured during the simulation.
        """
        # add some info to the data dictionary
        self.data['steps'] = num_steps
        self.data['beta'] = self.beta
        self.data['grid_size'] = self.grid_size
        self.sample_rate = sample_rate
        
        if observables != None: 
            for ob in observables:
                if ob.__name__ not in self.data:  # if key not in dict, set it to empty list
                    self.data[ob.__name__] = [] 
   
        for i in trange(num_steps, disable = not progress_bar):  
            # store data every sample_rate steps
            if i % sample_rate == 0 and observables != None:   
                for ob in observables:
                    self.data[ob.__name__].append(ob())

            # choose a random color 
            c = np.random.randint(0, self.num_colors, dtype=int)
         
            # choose a random square
            s = np.random.randint(1, self.grid_size+1, size = 2, dtype=int)

            # get num_link on each side of square s
            S = np.zeros(4, dtype=int)
            S[0] = self.grid[c, s[0], s[1], 0]
            S[2] = self.grid[c, s[0], s[1]-1, 0]
            S[1] = self.grid[c, s[0], s[1], 1]
            S[3] = self.grid[c, s[0]-1, s[1], 1]

            # get list of all possible transformation
            transformations = get_possible_transformations(S)                    ############################ MINIMAL OR FULL ST ####################################
            #transformations = self.minimal_transformations(S)
            
            # pick uniformly a random transformation
            M = len(transformations)   # num of possible transformation of current state, compute only once! we also need it to compute tha ratio M/M_prime in acceptance_prob
            index = np.random.randint(0, M)  
            X = transformations[index]
            
            '''
            if self.acceptance_prob(S, M, s, X, c) >= random.random():          (S, M, s, X, c, beta, num_colors, algo, get_possible_transformations, get_local_time, get_local_time_i)
                self.accepted += 1
                self.square_transformation(c, s, X)
            else:
                self.rejected += 1
            '''
            
            if acceptance_prob_optimized(S, M, s, X, c, self.beta, self.num_colors, self.algo, self.grid) >= random.random():  
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
          The current state of the grid.
        """
        return self.grid

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
        return np.max(self.grid, axis=(1,2,3))
    
    def avg_links(self):
        """
        Return the average number of links for each color.

        Returns:
          The average number of links for each color.
        """
        return np.mean(self.grid, axis = (1,2,3))

    def avg_local_time(self):
        """
        Return the average local time for the grid.

        Returns:
          The average local time for the grid.
        """
        total_local_time = 0
        for x in range(1,self.grid_size+1):
            for y in range(1,self.grid_size+1):
                total_local_time += self.get_local_time(x,y)
        return total_local_time / self.V
    
    def loop_builder(self, v1 = None, v2 = None):
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
        loops = []
        lenghts = []
        for c in range(self.num_colors):
            color_loops = []
            #copy the grid 
            G = np.copy(self.grid[c])
            nz = G.nonzero()
            non_zero = len(nz[0])
            if non_zero == 0:
                    return [], [0]
            
            while non_zero > 0:
                #pick first non-zero and unvisited edge
                nz = G.nonzero()
                non_zero = len(nz[0])
                if non_zero == 0:
                    break
                # choose a random vertex with non-zero incident links, or choose v1
                if v1 == None:
                    rand_index = np.random.randint(0, non_zero)
                    x, y  = nz[0][rand_index], nz[1][rand_index]
                else:
                    x, y = v1[0], v1[1]
                    
                starting_vertex = (x,y)
                current_loop = []
                
                # first step outside loops
                dir = []
                
                top = G[x,y+1,1]
                right = G[x+1,y,0]
                bottom = G[x,y,1]
                left = G[x,y,0]
                
                if top > 0:
                    dir.extend([0]*top)
                elif right > 0:
                    dir.extend([1]*right)
                elif bottom > 0:
                    dir.extend([2]*bottom)
                elif left > 0:
                    dir.extend([3]*left)
                
                if len(dir) == 0:
                    return 0 
                
                # pick a random dir with prob prop to num_links  
                rand_dir = np.random.choice(dir)
                
                match rand_dir:
                    case 0:
                        # remove one link
                        G[x,y+1,1] -= 1
                        #move there
                        y += 1
                    case 1:
                        G[x+1,y,0] -= 1
                        x += 1
                    case 2:
                        G[x,y,1] -= 1
                        y -= 1
                    case 3:
                        G[x,y,0] -= 1
                        x -= 1
                length = 0
                while True:
                    current_loop.append((x,y))
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
                    elif right > 0:
                        dir.extend([1]*right)
                    elif bottom > 0:
                        dir.extend([2]*bottom)
                    elif left > 0:
                        dir.extend([3]*left)

                    if (x,y) == starting_vertex:
                        if random.random() <= 1/(len(dir)+1):
                            lenghts.append(length)
                            break
                    
                    # pick a random dir with prob prop to num_links  
                    rand_dir = np.random.choice(dir)
                    
                    match rand_dir:
                        case 0:
                            # remove one link
                            G[x,y+1,1] -= 1
                            #move there
                            y += 1
                        case 1:
                            G[x+1,y,0] -= 1
                            x += 1
                        case 2:
                            G[x,y,1] -= 1
                            y -= 1
                        case 3:
                            G[x,y,0] -= 1
                            x -= 1
                        
                color_loops.append(current_loop)
                
            loops.append(color_loops)
        return loops, lenghts
    
    def avg_loop_length(self):
        """
        Return the average loop length for the grid.

        Returns:
          The average loop length for the grid.
        """
        _, lengths = self.loop_builder()
        return np.mean(lengths)
                
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
    def plot_one_color(self, c, cmap, ax, alpha = 1.0, linewidth = 1.0): # plots the grid of just one color
        """
        Plot the grid for a given color c.

        Args:
          c: The color to be plotted.
          cmap: The colormap to be used for plotting.
          ax: The axis on which to plot the grid.
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        for x in range(0, self.grid_size+2):
                for y in range(0, self.grid_size+2):
                    # horizontal
                    if self.grid[c,x,y,0] != 0:
                        edge_color = cmap(self.grid[c,x,y,0])
                        ax.plot([x-1, x], [y, y], color=edge_color, linewidth=linewidth, alpha = alpha)
                    # vertical
                    if self.grid[c,x,y,1] != 0:   
                        edge_color = cmap(self.grid[c,x,y,1])
                        ax.plot([x, x], [y, y-1], color=edge_color, linewidth=linewidth, alpha = alpha)
        
    def plot_loop(self, c, loop, color = 'yellow', alpha = 0.25, linewidth = 1.5): 
        """
        Highlights a loop in a given color c.

        Args:
          c: The color for which to plot.
          loop: The loop to be plotted.
          color: The color to be used for plotting the loop. Default is yellow
          alpha: The transparency level for the plot.
          linewidth: The width of the lines in the plot.
        """
        fig, ax = plt.subplots(figsize=(12,12))
        num_segments = int(self.max_links()[c]+1)            #color dependet!
        cmap = create_cmap(self.num_colors, c, num_segments)
        self.plot_one_color(c, cmap, ax)
        for i in range(len(loop)-1):
            ax.plot( [loop[i][0], loop[i+1][0]], [loop[i][1], loop[i+1][1]], linewidth=linewidth, color = color, alpha = alpha)
        #draw last link
        ax.plot( [loop[0][0], loop[-1][0]], [loop[0][1], loop[-1][1]], linewidth=1.5, color = color, alpha = alpha)
        ax.set_title('length = {}'.format(len(loop)))
        
    def plot_grid(self, figsize = (10,8), linewidth = 1.0, colorbar = True, file_name = None):
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
        fig, axes = plt.subplots(1,self.num_colors,figsize = figsize, gridspec_kw={'hspace': 0.05, 'wspace': 0.05}) #, facecolor='black')
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
            #axes[c].set_title('avg links = {}'.format(self.avg_links()[c]))
            axes[c].set_xlim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            axes[c].set_ylim(-(1+0.05*self.grid_size), 2+self.grid_size*1.05)
            #axes[c].axis('square')
            axes[c].axis('off')                                                ########### axis
            
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
        fig, ax = plt.subplots(figsize = figsize)

        for c in range(self.num_colors):
            # Define a colormap
            num_segments = int(self.max_links()[c]+1) if not normalized else 2
            cmap = create_cmap(self.num_colors, c, num_segments)
            
            # Create a ScalarMappable for colorbar
            norm = Normalize(vmin=0, vmax=num_segments)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Empty array, as we'll not use actual data

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
    
    def summary(self):
        """
        Print a summary of the current state of the grid.
        """
        print('average number of links: {}'.format(self.avg_links()))
        print('max number of links: {}'.format(self.max_links() ))
        print('avg local time: {}'.format(self.avg_local_time()))
        loops, lengths = self.loop_builder() #stats on color 0

        print('avg loop length: {}'.format(np.mean(lengths)))
        print('max loop length: {}'.format(max(lengths)))
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
        with open(file_name, 'w') as file:
            json.dump(self.data, file)
            
    def load_data(self, file_name):
        """
        Load the state of the grid from a file.

        Args:
          file_name: The name of the file to load the data from.
        """
        with open(file_name, 'r') as file:
            self.data = json.load(file)
            
            
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
    Generates n RGB colors in the form of (r, g, b) with r, g, b values in [0, 1].
    For n = 3, it returns red, green, and blue. For other values of n, it aims to maximize
    the perceptual difference between the colors, ensuring they are bright and vivid.
    """
    if n == 3:
        # Directly return red, green, blue for n = 3
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    
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



        
        
           
#########################################################################################################
#
#     To improve performance, we redefine computationally heavy functions outside and use numba 
#
#########################################################################################################

@jit(fastmath = True, nopython = True)
def get_possible_transformations(S):
    """
    Return a list of all possible transformations for a given square S.

    Args:
        S: The square for which to generate all possible transformations.

    Returns:
        A list of all possible transformations for the given square.
    """
    # list of all possible transformations

    transformations = [ 
        (-1,-1,-1,-1),                    # uniform                     
        (-2, 0, 0, 0),                    # single top
        (0, -2, 0, 0),                    # single right
        (0, 0, -2, 0),                    # single bottom
        (0, 0, 0, -2),                    # single left
        (1,-1, 1,-1), (-1, 1,-1, 1),      # swap opposite
        (1, 1,-1,-1), (-1,-1, 1, 1),      # swap adjacent
        (-1, 1,1,-1), ( 1,-1,-1, 1)#,   
        #(1,-1,-1,-1), (-1, 1, 1, 1),      # triple up
        #(-1,1,-1,-1), ( 1,-1, 1, 1),      # triple right
        #(-1,-1,1,-1), ( 1, 1,-1, 1),      # triple bottom
        # (-1,-1,-1,1), ( 1, 1, 1,-1)       # triple left
        ]    
    
    if S[0] < 2:                                                # top is 0 or 1, remove single top -2
        transformations.remove( (-2, 0, 0, 0) )
        if S[0] == 0:                                           # top is 0, remove uniform -1, swap 
            transformations.remove((-1, -1, -1, -1))
            transformations.remove((-1, 1, -1, 1))
            transformations.remove((-1, -1, 1, 1))
            transformations.remove((-1, 1, 1, -1))
            #transformations.remove((-1, 1, 1, 1))
            #transformations.remove((-1, 1, -1, -1))
            #transformations.remove((-1, -1, 1, -1))
            #transformations.remove((-1, -1, -1, 1))
    if S[1] < 2:
        transformations.remove((0, -2, 0, 0))
        if S[1] == 0:
            if (-1, -1,-1,-1) in transformations: transformations.remove((-1, -1,-1,-1))
            if (1, -1, 1, -1) in transformations: transformations.remove(( 1, -1, 1,-1))
            if (-1, -1, 1, 1) in transformations: transformations.remove((-1, -1, 1, 1))
            if (1, -1, -1, 1) in transformations: transformations.remove(( 1, -1,-1, 1))
            #if (1, -1, -1, -1) in transformations: transformations.remove((1, -1, -1, -1))
            #if (1, -1, 1, 1) in transformations: transformations.remove((1, -1, 1, 1))
            #if (-1, -1, 1, -1) in transformations: transformations.remove((-1, -1, 1, -1))
            #if (-1, -1, -1, 1) in transformations: transformations.remove((-1, -1, -1, 1))

    if S[2] < 2:
        transformations.remove((0, 0, -2, 0))
        if S[2] == 0:
            if (-1, -1,-1,-1) in transformations: transformations.remove((-1,-1, -1,-1))
            if (-1, 1, -1, 1) in transformations: transformations.remove((-1, 1, -1, 1))
            if (1, 1, -1, -1) in transformations: transformations.remove(( 1, 1, -1,-1))
            if (1, -1, -1, 1) in transformations: transformations.remove(( 1,-1, -1, 1))
            #if (1, -1, -1, -1) in transformations: transformations.remove((1, -1, -1, -1))
            #if (-1, 1, -1, -1) in transformations: transformations.remove((-1, 1, -1, -1))
            #if (1, 1, -1, 1) in transformations: transformations.remove((1, 1, -1, 1))
            #if (-1, -1, -1, 1) in transformations: transformations.remove((-1, -1, -1, 1))
    
    if S[3] < 2:
        transformations.remove((0, 0, 0, -2))
        if S[3] == 0:
            if (-1, -1,-1,-1) in transformations: transformations.remove((-1,-1,-1, -1))
            if (1, -1, 1, -1) in transformations: transformations.remove(( 1,-1, 1, -1))
            if (1, 1, -1, -1) in transformations: transformations.remove(( 1, 1,-1, -1))
            if (-1, 1, 1, -1) in transformations: transformations.remove((-1, 1, 1, -1))
            #if (1, -1, -1, -1) in transformations: transformations.remove((1, -1, -1, -1))
            #if (-1, 1, -1, -1) in transformations: transformations.remove((-1, 1, -1, -1))
            #if (-1, -1, 1, -1) in transformations: transformations.remove((-1, -1, 1, -1))
            #if (1, 1, 1, -1) in transformations: transformations.remove((1, 1, 1, -1))


    return transformations + [(1, 1, 1, 1), (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]   #add back always applicable transformations

@jit(fastmath = True, nopython = True)
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

@jit(fastmath = True, nopython = True)
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
                       
@jit(fastmath = True, nopython = True)
def acceptance_prob_optimized(S, M, s, X, c, beta, num_colors, algo, grid):
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
    M_prime = len(get_possible_transformations(S_p))
    A = 0
    num_colors_half = num_colors / 2

    if np.array_equal(X, [1, 1, 1, 1]):
        A = beta**4 / (16 * S_p[0]*S_p[1]*S_p[2]*S_p[3]) * \
            (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1)) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1)) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1)
    elif np.array_equal(X, [-1, -1, -1, -1]):
        A = (16 / (beta**4)) * S[0]*S[1]*S[2]*S[3] * \
            (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) - 1))
    elif np.array_equal(X, [2, 0, 0, 0]):
        A = beta**2 / (4 * S_p[0] * (S[0]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]))) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1)
    elif np.array_equal(X, [-2, 0, 0, 0]):
        A = 4 * S[0] * (S[0]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1])))
    elif np.array_equal(X, [0, 2, 0, 0]):
        A = beta**2 / (4 * S_p[1] * (S[1]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1))) * \
            (2*get_local_time_i(grid, c, s[0], s[1]) + 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1)
    elif np.array_equal(X, [0, -2, 0, 0]):
        A = 4 * S[1] * (S[1]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0], s[1]) - 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1)))
    elif np.array_equal(X, [0, 0, 2, 0]):
        A = beta**2 / (4 * S_p[2] * (S[2]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1)) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1))) * \
            (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1) + 1)
    elif np.array_equal(X, [0, 0, -2, 0]):
        A = 4 * S[2] * (S[2]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0], s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0]-1, s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0], s[1]-1)))
    elif np.array_equal(X, [0, 0, 0, 2]):
        A = beta**2 / (4 * S_p[3] * (S[3]+1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1])) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1))) * \
            (2*get_local_time_i(grid, c, s[0]-1, s[1]-1) + 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1]) + 1)
    elif np.array_equal(X, [0, 0, 0, -2]):
        A = 4 * S[3] * (S[3]-1) / (beta**2) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]) - 1) * (num_colors_half + get_local_time(grid, num_colors,s[0]-1, s[1]-1) - 1) / \
            ((2*get_local_time_i(grid, c, s[0]-1, s[1]-1) - 1) * (2*get_local_time_i(grid, c, s[0]-1, s[1])))
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
    return min(1, M/M_prime * A) if algo == 'metropolis' else 1/(1 + M_prime/(M*A))