import cProfile
import pstats
from RandomLoop import StateSpace



def main():
    """Profiles the defined functions."""
    m = StateSpace(3, 64, 2)
    profiler = cProfile.Profile()
    
    # List of functions to profile
    functions_to_test = [
        m.mean_links,
        m.loop_builder,
        m.loop_builder_fast,
        m.check_percolation
    ]

    profiler.enable()    
    # 1M steps 
    m.step(100_000)
    # Call each function and disable profiler after each call
    for func in functions_to_test:
        func()
    profiler.disable()
    
    names = [f.__name__ for f in functions_to_test]
    
    # Print results for the profiled function
    stats = pstats.Stats(profiler)
    stats.print_stats(names + [m.step.__name__])
    
  

def print_stats(stats):
    """Prints function name and average time per call."""
    print("Function Name          | Average Time (s)")
    print("-----------------------|------------------")
    for entry in stats:
        # Access function name from the first element of the tuple (index 0)
        name = format_function_name(entry[0])
        total_time = entry[1]
        print(entry)
        # Ensure ncalls is numeric before division (handle 'cumulative' sort)
        try:
            ncalls = int(entry[2])
        except ValueError:
            ncalls = 0
        if ncalls > 0:
            avg_time = total_time / ncalls
        print(f"{name:<25s} | {avg_time:.4f}")


def format_function_name(name):
  """Formats function names for nicer output."""
  return name.split("(")[0].rsplit(".", 1)[-1]  # Extract function name (optional)


if __name__ == "__main__":
  main()
