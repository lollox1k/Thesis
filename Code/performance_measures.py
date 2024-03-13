import cProfile
import pstats
import numpy as np
from RandomLoop import StateSpace, get_possibile_transformations, minimal_transformations  # Import your functions

profiler = cProfile.Profile()

m = StateSpace(3, 32, 5.0)

m.step()

profiler.enable()

m.step(1_000)

profiler.disable()

# Correct way to sort and print stats
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()

# If you want to specifically filter for your functions, you might do something like:
stats.print_stats(m.step.__name__)

