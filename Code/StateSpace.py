import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from matplotlib import cm
import colorsys
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap, PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import random


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

class stateSpace:
    def __init__(self, num_colors, grid_size, beta, init = 0, bc = 0, algo = 'metropolis'):  # 'glauber'      
    
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

        if init == 'random':
            self.random_init()
            
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
        
    def step(self, num_steps = 1, progress_bar = True, sample_rate = 10_000, observables = None):   # observables is a list of functions like [m.avg_links, m.avg_local_time]
        
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
            transformations = self.get_possible_transformations(S)                    ############################ MINIMAL OR FULL ST ####################################
            #transformations = self.minimal_transformations(S)
            
            # pick uniformly a random transformation
            M = len(transformations)   # num of possible transformation of current state, compute only once! we also need it to compute tha ratio M/M_prime in acceptance_prob
            index = np.random.randint(0, M)  
            X = transformations[index]
            #print('transformation: {}'.format(X))
            
            if self.acceptance_prob(S, M, s, X, c) >= random.random():
                self.accepted += 1
                self.square_transformation(c, s, X)
            else:
                self.rejected += 1
    
    def minimal_transformations(self, S):
        # list of just minimal transformations for irreducibility

        transformations = [ 
            (1, 1, 1, 1),                      # uniform +1
            (-2, 0, 0, 0),                    # single top
            (0, -2, 0, 0),                    # single right
            (0, 0, -2, 0),                    # single bottom
            (0, 0, 0, -2)                    # single left
            ]    
        
        if S[0] < 2:                                                # top is 0 or 1, remove single top -2
            transformations.remove( (-2, 0, 0, 0) )
            
        if S[1] < 2:
            transformations.remove((0, -2, 0, 0))
           
        if S[2] < 2:
            transformations.remove((0, 0, -2, 0))
        
        if S[3] < 2:
            transformations.remove((0, 0, 0, -2))

        return transformations 
    
    def get_possible_transformations(self, S):
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

    def square_transformation(self, c, s, X):    # X = (a_1, a_2, a_3, a_4) like in the thesis
        #top
        self.grid[c, s[0], s[1], 0] += X[0]
        #right
        self.grid[c, s[0], s[1], 1] += X[1]
        #bottom
        self.grid[c, s[0], s[1]-1, 0] += X[2]
        #left
        self.grid[c, s[0]-1, s[1], 1] += X[3]
    
    def acceptance_prob(self, S, M, s, X, c): # S = (l1,l2,l3,l4) links on the square, M, s = (x,y) vertex in the grid, X = (a_1,a_2,a_3,a_4) square transformation, c = 1,..., self.num_colors color of the transformation
        # possible transformation ratio
        S_prime = np.copy(S) 
        S_prime += np.array(X) #apply the transformation in the square

        #get M_prime, the number of possibile transformation of the new state
        M_prime = len(self.get_possible_transformations(S_prime))                                 ############################ MINIMAL OR FULL ST ####################################
        #M_prime = len(self.minimal_transformations(S_prime))

        # prob ratio
        match X:
            case (1, 1, 1, 1):                                                               # '\' is used to split lines when they are too long!
                A = self.beta**4 / (16 * (S[0] + 1)*(S[1] + 1)*(S[2] + 1)*(S[3] + 1) \
                *(self.num_colors/2 + self.get_local_time(s[0], s[1]))*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1))*(self.num_colors/2 + self.get_local_time(s[0]-1 , s[1]))*(self.num_colors/2 + self.get_local_time(s[0]-1 , s[1]-1))) \
                *(2*self.get_local_time_i(c, s[0], s[1]) + 1)*(2*self.get_local_time_i(c, s[0], s[1]-1) + 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]-1) + 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]) + 1)
                
            case (-1,-1,-1,-1):
                A = (16 / (self.beta**4)) * S[0]*S[1]*S[2]*S[3] *(self.num_colors/2 + self.get_local_time(s[0], s[1]) - 1)*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1 ) - 1)*(self.num_colors/2 + self.get_local_time(s[0]-1, s[1]) - 1)*(self.num_colors/2 + self.get_local_time(s[0]-1, s[1]-1) - 1) \
                    / ( (2*self.get_local_time_i(c, s[0], s[1]) - 1)*(2*self.get_local_time_i(c, s[0], s[1]-1) - 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]-1) - 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]) - 1) )
            
            case (2, 0, 0, 0):
                A = self.beta**2/ (4*(S[0]+2)*(S[0]+1)*(self.num_colors/2 + self.get_local_time(s[0], s[1]))*(self.num_colors/2 + self.get_local_time( s[0]-1, s[1]))  ) \
                    * (2*self.get_local_time_i(c, s[0], s[1]) + 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]) + 1)
                    
            case (-2, 0, 0, 0):
                A = 4 * S[0]*(S[0]-1) / self.beta**2 * (self.num_colors/2 + self.get_local_time(s[0], s[1]) - 1)*(self.num_colors/2 + self.get_local_time(s[0]-1, s[1]) - 1) \
                    / ( (2*self.get_local_time_i(c, s[0], s[1])- 1)*(2*self.get_local_time_i(c, s[0]-1, s[1])) )
                    
            case (0, 2, 0, 0):
                A = self.beta**2/ (4*(S[1]+2)*(S[1]+1)*(self.num_colors/2 + self.get_local_time(s[0], s[1]))*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1)) )  \
                    * (2*self.get_local_time_i(c, s[0], s[1]) + 1)*(2*self.get_local_time_i(c, s[0], s[1]-1) + 1)
                
            case (0,-2, 0, 0): 
                A = 4 * S[1]*(S[1]-1) / self.beta**2 * (self.num_colors/2 + self.get_local_time(s[0], s[1]) - 1)*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1) - 1) \
                    / ( (2*self.get_local_time_i(c, s[0], s[1])- 1)*(2*self.get_local_time_i(c, s[0], s[1]-1)) )
            case (0, 0, 2, 0):
                A = self.beta**2/ (4*(S[2]+2)*(S[2]+1)*(self.num_colors/2 + self.get_local_time( s[0]-1, s[1]-1))*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1))  ) \
                     * (2*self.get_local_time_i(c, s[0]-1, s[1]-1) + 1)*(2*self.get_local_time_i(c, s[0], s[1]-1) + 1)
            case (0, 0,-2, 0):
                A = 4 * S[2]*(S[2]-1) / self.beta**2 * (self.num_colors/2 + self.get_local_time( s[0]-1, s[1]-1 ) - 1)*(self.num_colors/2 + self.get_local_time(s[0], s[1]-1 ) - 1) \
                    / ( (2*self.get_local_time_i(c, s[0]-1, s[1]-1)- 1)*(2*self.get_local_time_i(c, s[0], s[1]-1)) )
            case (0, 0, 0, 2):
                A = self.beta**2/ (4*(S[3]+2)*(S[3]+1)*(self.num_colors/2 + self.get_local_time( s[0]-1, s[1]))*(self.num_colors/2 + self.get_local_time( s[0]-1, s[1]-1))  ) \
                    * (2*self.get_local_time_i(c, s[0]-1, s[1]-1) + 1)*(2*self.get_local_time_i(c, s[0]-1, s[1]) + 1)
            case (0, 0, 0,-2):
                A = 4 * S[3]*(S[3]-1) / self.beta**2 * (self.num_colors/2 + self.get_local_time( s[0]-1, s[1]) - 1)*(self.num_colors/2 + self.get_local_time( s[0]-1, s[1]-1) - 1) \
                    / ( (2*self.get_local_time_i(c, s[0]-1, s[1]-1)- 1)*(2*self.get_local_time_i(c, s[0]-1, s[1])) )
            
            case (-1, 1, -1, 1):
                A = S[0]*S[2]/( (S[1]+1)*(S[3]+1) )
            case (1, -1, 1, -1):
                A = S[1]*S[3]/( (S[0]+1)*(S[2]+1) )
                
            case (-1, -1, 1, 1):
                A = S[0]*S[1] / ( (S[2]+1)*(S[3]+1) ) * (self.num_colors/2 + self.get_local_time(s[0], s[1]) - 1 ) / (self.num_colors/2 + self.get_local_time(s[0]-1, s[1]-1)) \
                    * (2*self.get_local_time_i(c, s[0]-1, s[1]-1)+ 1)/(2*self.get_local_time_i(c, s[0], s[1]) - 1)
            case (1, 1, -1, -1):
                A = S[2]*S[3] / ( (S[0]+1)*(S[1]+1) ) * (self.num_colors/2 + self.get_local_time(s[0]-1, s[1]-1) -1 ) / (self.num_colors/2 + self.get_local_time(s[0], s[1]))  \
                    * (2*self.get_local_time_i(c, s[0], s[1])+ 1)/(2*self.get_local_time_i(c, s[0]-1, s[1]-1) - 1)
            case (-1, 1, 1, -1):
                A = S[0]*S[3] / ( (S[1]+1)*(S[2]+1) ) * (self.num_colors/2 + self.get_local_time(s[0]-1, s[1]) -1 ) / (self.num_colors/2 + self.get_local_time(s[0], s[1]-1)) \
                    * (2*self.get_local_time_i(c, s[0], s[1]-1)+ 1)/(2*self.get_local_time_i(c, s[0]-1, s[1]) - 1)
            case (1, -1, -1, 1):
                A = S[2]*S[1] / ( (S[3]+1)*(S[0]+1) ) * (self.num_colors/2 + self.get_local_time(s[0], s[1]-1) -1 ) / (self.num_colors/2 + self.get_local_time(s[0]-1, s[1]) ) \
                    * (2*self.get_local_time_i(c, s[0]-1, s[1])+ 1)/(2*self.get_local_time_i(c, s[0], s[1]-1) - 1)

            # missing triple transformations

        #print('acceptance prob = {}'.format(min(1, M/M_prime * A)))
        return min(1, M/M_prime * A) if self.algo == 'metropolis' else 1/(1 + M_prime/(M*A))   # Metropolis  Glauber       #### May impact performanca a bit, better to edit the code!
    
    def get_grid(self):
        return self.grid
    
    def get_local_time(self, x, y):   # we know already the number of links in square s! we are wasting a bit of compute power 
        local_time = 0
        for c in range(self.num_colors):
            local_time += self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0]
        return local_time // 2
    
    def get_local_time_i(self, c, x, y): # returns local time of color c
        return (self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0] ) // 2

    def max_links(self): # improve
        return np.max(self.grid, axis=(1,2,3))
    
    def avg_links(self):  # None -> tuple of floats   returns the avg links for each color 
        return np.mean(self.grid, axis = (1,2,3))

    def avg_local_time(self):
        total_local_time = 0
        for x in range(1,self.grid_size+1):
            for y in range(1,self.grid_size+1):
                total_local_time += self.get_local_time(x,y)
        return total_local_time / self.V
    
    def loop_builder(self, v1 = None, v2 = None):  # given a link configuration, randomly builds loops (we are choosing a link pairing uniformly) returns list of loops (as a sequence of vertices) for each color
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
        _, lengths = self.loop_builder()
        return np.mean(lengths)
                
    def check_state(self): # checks if the current state m is legal (every vertex has even degree)
        for c in range(self.num_colors):
            for x in range(1,self.grid_size+1):
                for y in range(1,self.grid_size+1):
                    if (self.grid[c, x, y, 0] + self.grid[c, x, y, 1] + self.grid[c, x, y + 1, 1] + self.grid[c, x + 1, y, 0]) % 2 != 0:
                        print('###  illegal state!  ###')
                        return False 
        return True 
    def plot_one_color(self, c, cmap, ax, alpha = 1.0, linewidth = 1.0): # plots the grid of just one color
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
        
    def plot_loop(self, c, loop, color = 'yellow', alpha = 0.25, linewidth = 1.5):  # plots the longest loop over the grid
        fig, ax = plt.subplots(figsize=(12,12))
        num_segments = int(self.max_links()[c]+1)            #color dependet!
        cmap = create_cmap(c, num_segments)
        self.plot_one_color(c, cmap, ax)
        for i in range(len(loop)-1):
            ax.plot( [loop[i][0], loop[i+1][0]], [loop[i][1], loop[i+1][1]], linewidth=linewidth, color = color, alpha = alpha)
        #draw last link
        ax.plot( [loop[0][0], loop[-1][0]], [loop[0][1], loop[-1][1]], linewidth=1.5, color = color, alpha = alpha)
        ax.set_title('length = {}'.format(len(loop)))
        
    def plot_grid(self, figsize = (10,8), linewidth = 1.0, colorbar = True, file_name = None):     # plots the grid of every color 
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

    def plot_overlap(self, figsize = (12,12), normalized = False, file_name = None, alpha = 0.7, linewidth = 1.0):  # plots every color overlapped
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
    
    def summary(self):   # prints some stats
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
        save the data dictionary to a json file.
        """
        with open(file_name, 'w') as file:
            json.dump(self.data, file)
            
    def load_data(self, file_name):
        """
        load the data dictionary from a json file
        """
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        
            
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


def build_donut(m):
    #building the donut state
    for c in range(m.num_colors):
        for x in range(1, m.grid_size // 4+1):
            for y in range(1, m.grid_size+1):
                m.grid[c, x, y, 0] = 1
                m.grid[c, x, y, 1] = 1
        for x in range(m.grid_size+1 - m.grid_size // 4, m.grid_size+1):
            for y in range(1, m.grid_size+1):
                m.grid[c, x, y, 0] = 1
                m.grid[c, x, y, 1] = 1
    for c in range(m.num_colors):
        for y in range(1, m.grid_size // 4+1):
            for x in range(1, m.grid_size+1):
                m.grid[c, x, y, 0] = 1
                m.grid[c, x, y, 1] = 1
        for y in range(m.grid_size+1 - m.grid_size // 4, m.grid_size + 1):
            for x in range(1, m.grid_size+1):
                m.grid[c, x, y, 0] = 1
                m.grid[c, x, y, 1] = 1

    # add missing links on the outside border
    for c in range(m.num_colors):
        for y in range(1, m.grid_size+1):
            m.grid[c,0, y, 1] = 1
            m.grid[c, y, 0, 0] = 1 
            
    # add missing links on the inside border
    for c in range(m.num_colors):
        for x in range(m.grid_size // 4, 3 * m.grid_size // 4 + 1):
            m.grid[c,x,3 * m.grid_size // 4, 0] = 1
            m.grid[c, 3 * m.grid_size // 4, x, 1] = 1 
            
    #make it a legal state
    m.grid *= 2
    
    
def build_snake(m):
    # build concentric squares
    offset = m.grid_size // 2+1
    for c in range(m.num_colors):
        for k in range(1,offset):
            m.grid[c, k:-k, k-1, 0] = 1
            m.grid[c, k-1, k:-k, 1] = 1
            
            m.grid[c, (offset-k):(k-offset), offset + k -1, 0] = 1
            m.grid[c, offset + k -1, (offset-k):(k-offset), 1] = 1
            
    # swap some links to join the squares
    for c in range(m.num_colors):
        for k in range(1,offset-1):
            m.grid[c, k+1 , m.grid_size-k+1, 0] = 0
            m.grid[c, k+1 , m.grid_size-k, 0] = 0
            
            m.grid[c, k , m.grid_size-k+1, 1] = 1
            m.grid[c, k+1 , m.grid_size-k+1, 1] = 1