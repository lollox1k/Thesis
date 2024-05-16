import cProfile
import pstats
import numpy as np
from RandomLoop import StateSpace  # Import your functions


profiler = cProfile.Profile()

m = StateSpace(3, 128, 3)

profiler.enable()

for _ in range(100):
    m.random_init()
    m.loop_builder((64,64))
    m.loop_builder_fast((64,64))

profiler.disable()

# sort and print stats
stats = pstats.Stats(profiler).sort_stats('cumulative')
#stats.print_stats()

# filter functions
stats.print_stats(m.loop_builder.__name__)

