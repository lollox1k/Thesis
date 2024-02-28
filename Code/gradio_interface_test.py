# importing libraries
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

# import custom class
from RandomLoop import stateSpace, create_cmap

#plt.style.use("dark_background")
plt.style.use("dark_background")

def generate_plot(num_colors, grid_size, beta, steps):
    # run the simulation
    m = stateSpace(num_colors, grid_size, beta)
    m.step(steps)
    
    # plot it
    plots = []
    
    for c in range(num_colors):
        fig, ax = plt.subplots(figsize = (16,12))
        # create colormap
        num_segments = int(m.max_links()[c]+1)
        cmap = create_cmap(m.num_colors, c, num_segments)
        norm = Normalize(vmin=0, vmax=num_segments)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        
        # add colorbar
        cbar = plt.colorbar(sm, ax = ax)
        cbar.set_ticks(  0.5 + np.arange(0, num_segments,1))
        cbar.set_ticklabels(list(range(0, num_segments)))
        ax.axis('off')
        m.plot_one_color(c, cmap, ax) 
        
        # Convert the plot to an image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plots.append(plot_img)

    return plots

from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from matplotlib.collections import LineCollection



def plot_one_optimized(grid, c, cmap, ax, alpha=1.0):
    """
    Optimized function to plot grid lines using batch drawing with LineCollection for efficiency.
    """
    # Initialize lists to collect line segments
    segments = []

    # Collect line segments for horizontal and vertical lines
    for x in range(len(grid[0][0])):
        for y in range(len(grid[0][0])):
            # horizontal lines
            if grid[c][x][y][0] != 0:
                segments.append([(x - 1, y), (x, y)])
            # vertical lines
            if grid[c][x][y][1] != 0:
                segments.append([(x, y), (x, y - 1)])

    # Assuming colors are normalized between 0 and 1, adjust as needed
    line_colors = [cmap(grid[c][x][y][z]) for x in range(len(grid[0][0])) for y in range(len(grid[0][0])) for z in range(2) if grid[c][x][y][z] != 0]

    # Create a LineCollection
    lc = LineCollection(segments, colors=line_colors, linewidths=1.5, alpha=alpha)
    ax.add_collection(lc)
    return lc

# Placeholder for the animation's frame update function
def get_frame(i, ax, m, max_links):
    # Placeholder for grid_size and other data
    # Example grid size, replace with actual value
    
    artists = []
    ax.clear()
    
    for c in range(m.num_colors):
        # Placeholder for cmap creation logic
        cmap = create_cmap(m.num_colors, c, max_links[c]) 
        lc = plot_one_optimized(m.data['get_grid'][i], c, cmap, ax, 0.6)
        artists.append(lc)
        
    ax.set_xlim(-(1 + 0.05 * m.grid_size), 2 + m.grid_size * 1.05)
    ax.set_ylim(-(1 + 0.05 * m.grid_size), 2 + m.grid_size * 1.05)
    ax.axis('off')

    return artists

def generate_animation(num_colors, grid_size, beta, steps, sample_rate, normalize, initial_state, theme):
    if theme == 'dark':
        plt.style.use("dark_background")    
    elif theme == 'light':
        plt.style.use("default")

    # run it 
    if initial_state == 'zero':
        initial_state = 0
    m = stateSpace(num_colors, grid_size, beta, init=initial_state)
        
    m.step(num_steps=steps, sample_rate=sample_rate, observables=[m.get_grid])
    
    max_links = np.max(m.data['get_grid'], axis = (0, 2, 3, 4)) if not normalize else [2,2,2]
    
    fig, ax = plt.subplots(figsize=(16, 16))

    # adjust the range and interval based on your data and desired animation speed
    animation = FuncAnimation(fig, get_frame, frames=len(m.data['get_grid']), repeat=False, blit = True, fargs = [ax, m, max_links])

    # Display the animation in the notebook
    return animation.to_html5_video()


# Create the interface
demo = gr.Interface(fn=generate_animation, 
                     inputs=[
                         gr.Number(value = 3, minimum=2, maximum=4, step = 1,label = "number of colors"),
                         gr.Number(value = 32, minimum=4, maximum=128, step=1, label = "grid size"),
                         gr.Number(value = 1, minimum=0, maximum=128, step=1, label = "beta"),     
                         gr.Slider(value = 10_000, minimum=0, maximum=1_000_000, step=10_000, label = "steps"),
                         gr.Slider(value = 1_000, minimum=0, maximum=100_000, step=1_000, label = "sample rate"),
                         gr.Checkbox(value = True, label = "Normalize color"),
                         gr.Dropdown(["zero", "random", "snake", "donut"], value = "zero", label="Initial state"),
                         gr.Radio(["dark", "light"], value = "dark", label="Theme")
                     ], 
                     outputs=gr.HTML(),
                     title="Simulation",
                     description="Enter parameters and click submit to start",
                     allow_flagging= "never"
                    )


'''

with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
    btn = gr.Button("Simulate", scale=0)

    btn.click(generate_plot, inputs = [
                         gr.Number(value = 3, minimum=2, maximum=4, step = 1,label = "number of colors"),
                         gr.Number(value = 32, minimum=4, maximum=128, step=1, label = "grid size"),
                         gr.Number(value = 1, minimum=0, maximum=128, step=1, label = "beta"),     
                         gr.Slider(value = 100_000, minimum=0, maximum=1_000_000, step=10_000, label = "steps")
                     ], outputs=gr.outputs.HTML(label="Model Output Video"))
'''


# Launch the interface
demo.launch(share = True)
