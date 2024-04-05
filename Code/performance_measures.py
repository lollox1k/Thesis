import cProfile
import pstats
import numpy as np
from RandomLoop import StateSpace  # Import your functions

profiler = cProfile.Profile()

m = StateSpace(3, 64, 3)



profiler.enable()

for _ in range(100):
    m.random_init()
    m.loop_builder((32,32))

profiler.disable()

# sort and print stats
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()

# filter functions
stats.print_stats(m.loop_builder.__name__)
stats.print_stats(np.copy.__name__)


