import tkinter as tk

class CycleDrawerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Cycle Drawer")
        self.grid_size = 10
        self.cell_size = 500 / self.grid_size
        self.cycle_edges = []
        self.preview_line = None

        self.canvas = tk.Canvas(self.master, width=500, height=500, bg="white")
        self.canvas.pack()

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.add_edge)
        self.canvas.bind("<Button-3>", self.undo_edge)
        self.canvas.bind("<Motion>", self.preview_edge)
        self.master.bind("<Key>", self.handle_key)
        self.quit_button = tk.Button(self.master, text="Generate TikZ Code", command=self.generate_tikz_code)
        self.quit_button.pack()

    def draw_grid(self):
        for i in range(self.grid_size + 1):
            self.canvas.create_line(0, i * self.cell_size, 500, i * self.cell_size, fill="gray", dash=(2, 2))
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, 500, fill="gray", dash=(2, 2))

    def add_edge(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = int(x / self.cell_size), int(y / self.cell_size)

        if len(self.cycle_edges) > 0:
            prev_x, prev_y = self.cycle_edges[-1]
            # Use the grid coordinates directly
            self.cycle_edges.append((grid_x, grid_y))
            self.draw_vertex(grid_x, grid_y, "blue")
            self.canvas.create_line(prev_x * self.cell_size, prev_y * self.cell_size,
                                     grid_x * self.cell_size, grid_y * self.cell_size,
                                     fill="red", width=2)
        else:
            self.cycle_edges.append((grid_x, grid_y))
            self.draw_vertex(grid_x, grid_y, "blue")

    def undo_edge(self, event):
        if len(self.cycle_edges) > 0:
            last_edge = self.cycle_edges.pop()
            self.canvas.delete("vertex")
            self.draw_vertices(self.cycle_edges)
            if len(self.cycle_edges) > 0:
                self.draw_vertex(self.cycle_edges[-1][0], self.cycle_edges[-1][1], "blue")

    def preview_edge(self, event):
        x, y = event.x, event.y

        if len(self.cycle_edges) > 0:
            prev_x, prev_y = self.cycle_edges[-1]

            # Remove previous preview line
            if self.preview_line:
                self.canvas.delete(self.preview_line)

            # Draw the new preview line
            self.preview_line = self.canvas.create_line(prev_x * self.cell_size, prev_y * self.cell_size,
                                                         int(x / self.cell_size) * self.cell_size,
                                                         int(y / self.cell_size) * self.cell_size,
                                                         fill="blue", width=2)

    def draw_vertex(self, x, y, color):
        radius = 10
        self.canvas.create_oval(x * self.cell_size - radius, y * self.cell_size - radius,
                                x * self.cell_size + radius, y * self.cell_size + radius,
                                fill=color, outline="", tags="vertex")

    def draw_vertices(self, vertices):
        for vertex in vertices:
            self.draw_vertex(vertex[0], vertex[1], "blue")

    def generate_tikz_code(self):
        tikz_code = "\\begin{tikzpicture}\n"
        tikz_code += f"    % Draw grid\n    \\draw[step=0.5, gray, thin] (0,0) grid ({self.grid_size},{self.grid_size});\n"
        tikz_code += "    % Draw closed cycle (red boundary)\n    \\draw[red, thick]\n"

        for edge in self.cycle_edges:
            tikz_code += f"        ({edge[0]},{edge[1]}) --"

        # Close the cycle
        tikz_code += f" cycle;\n"
        tikz_code += "\\end{tikzpicture}"

        print(tikz_code)

    def handle_key(self, event):
        if event.char == 'R':
            self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("vertex")
        self.canvas.delete("preview_line")
        self.cycle_edges = []

    def __del__(self):
        # Cleanup, remove the preview line
        if self.preview_line:
            self.canvas.delete(self.preview_line)


def main():
    root = tk.Tk()
    app = CycleDrawerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
