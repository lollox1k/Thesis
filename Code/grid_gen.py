def generate_tikz_code(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    tikz_code = "\\begin{tikzpicture}[scale=2]\n"
    
    # Grid lines with edge labels
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                tikz_code += f"\\draw ({j},{i}) -- ({j+1},{i}) node[midway,below] {{{matrix[i][j]}}};\n"  # Horizontal edges
            if i < rows - 1:
                tikz_code += f"\\draw ({j},{i}) -- ({j},{i+1}) node[midway,left] {{{matrix[i][j]}}};\n"   # Vertical edges
    
    # Adding top and left edges
    tikz_code += "\\draw (0,0) -- (0,{});\n".format(rows)
    tikz_code += "\\draw (0,0) -- ({},0);\n".format(cols)

    # Circles at each intersection
    for i in range(cols):
        for j in range(rows):
            tikz_code += f"\\filldraw ({i},{j}) circle (0.5pt);\n"  # Reduced circle size to 0.5pt
    
    tikz_code += "\\end{tikzpicture}"
    
    return tikz_code

# Example usage
matrix = [[0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [20, 21, 22, 23, 24]]

tikz_code = generate_tikz_code(matrix)
print(tikz_code)
