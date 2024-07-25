import pygame
import numpy as np

# Constants
WIDTH, HEIGHT = 1200, 900
CELL_SIZE = 10
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Conway's Game of Life")

def create_empty_grid():
    return np.zeros((ROWS, COLS))

grid = create_empty_grid()

def count_neighbors(grid, row, col):
    neighbors = 0
    for i in range(max(0, row-1), min(ROWS, row+2)):
        for j in range(max(0, col-1), min(COLS, col+2)):
            if (i, j) != (row, col):
                neighbors += grid[i, j]
    return neighbors

def update_grid(grid):
    new_grid = grid.copy()
    for row in range(ROWS):
        for col in range(COLS):
            neighbors = count_neighbors(grid, row, col)
            if grid[row, col] == 1:
                if neighbors < 2 or neighbors > 3:
                    new_grid[row, col] = 0
            else:
                if neighbors == 3:
                    new_grid[row, col] = 1
    return new_grid

def draw_grid(grid):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if grid[row, col] == 1 else BLACK
            pygame.draw.rect(screen, color, (col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_grid_overlay():
    for row in range(ROWS):
        pygame.draw.line(screen, GRAY, (0, row*CELL_SIZE), (WIDTH, row*CELL_SIZE))
    for col in range(COLS):
        pygame.draw.line(screen, GRAY, (col*CELL_SIZE, 0), (col*CELL_SIZE, HEIGHT))

# Game loop
running = True
drawing = True
mouse_down_left = False
mouse_down_right = False
show_grid_overlay = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and drawing:
            if event.button == 1:  # Left mouse button
                mouse_down_left = True
            elif event.button == 3:  # Right mouse button
                mouse_down_right = True
        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            if event.button == 1:  # Left mouse button
                mouse_down_left = False
            elif event.button == 3:  # Right mouse button
                mouse_down_right = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                drawing = False
            elif event.key == pygame.K_r:
                grid = create_empty_grid()
                drawing = True
            elif event.key == pygame.K_g:
                show_grid_overlay = not show_grid_overlay

    if drawing:
        if mouse_down_left or mouse_down_right:
            col, row = pygame.mouse.get_pos()
            col //= CELL_SIZE
            row //= CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                if mouse_down_left:
                    grid[row, col] = 1
                elif mouse_down_right:
                    grid[row, col] = 0

        screen.fill(BLACK)
        draw_grid(grid)
        if show_grid_overlay:
            draw_grid_overlay()
        pygame.display.flip()
    else:
        screen.fill(BLACK)
        draw_grid(grid)
        grid = update_grid(grid)
        pygame.display.flip()
        pygame.time.delay(100)

pygame.quit()