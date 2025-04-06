import pygame

pygame.init()

# Game constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 8
GRID_SIZE_LITTLE = 6
CELL_SIZE = 40
GRID_OFFSET_X = (WINDOW_WIDTH - GRID_SIZE * CELL_SIZE) // 2
GRID_OFFSET_Y = 100
BLOCK_TYPES = [
    [[1],[1],[1]], 
    [[1,1,1],[1],[1]],
    [[1],[1]],
    [[1],[1],[1]],
    [[1],[1],[1]],
    [[1],[1],[1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0], [1, 1]],
    [[0, 1], [1, 1]],
    [[1, 1], [1, 0]],
    [[1, 1], [0, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 0], [1,0,0]],
    [[1, 0], [0, 1]],
    [[0, 1], [1, 0]],
]
COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 165, 0)    # Orange
]
HIGHLIGHT_COLOR = (50, 200, 50, 128)  # Semi-transparent green for valid placement
INVALID_COLOR = (200, 50, 50, 128)    # Semi-transparent red for invalid placement
BACKGROUND_COLOR = (30, 30, 30)       # Dark background color
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game setup
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Wood Blocks')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)
title_font = pygame.font.SysFont('Arial', 36)


# GRIDS
# Empty grid