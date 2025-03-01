import pygame
from constants import CELL_SIZE, screen

class Block:
    def __init__(self, block_type, color, position=(0, 0)):
        self.shape = block_type
        self.color = color
        self.position = position  # Position in pixels
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self.width = len(self.shape[0]) * CELL_SIZE
        self.height = len(self.shape) * CELL_SIZE
        self.original_position = position
    
    def draw(self):
        for row_idx, row in enumerate(self.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    cell_rect = pygame.Rect(
                        self.position[0] + col_idx * CELL_SIZE,
                        self.position[1] + row_idx * CELL_SIZE,
                        CELL_SIZE, CELL_SIZE
                    )
                    pygame.draw.rect(screen, self.color, cell_rect)
                    pygame.draw.rect(screen, (0, 0, 0), cell_rect, 1)  # Border
    
    def is_point_inside(self, pos):
        x, y = pos
        block_x, block_y = self.position
        
        # Check if the point is within the block's bounding box
        if not (block_x <= x <= block_x + self.width and
                block_y <= y <= block_y + self.height):
            return False
        
        # Check if the point is on a filled cell
        cell_x = (x - block_x) // CELL_SIZE
        cell_y = (y - block_y) // CELL_SIZE
        
        if (0 <= cell_y < len(self.shape) and 
            0 <= cell_x < len(self.shape[0]) and 
            self.shape[cell_y][cell_x]):
            return True
        
        return False
    
    def start_drag(self, pos):
        self.dragging = True
        self.offset_x = self.position[0] - pos[0]
        self.offset_y = self.position[1] - pos[1]
    
    def update_position(self, pos):
        if self.dragging:
            self.position = (pos[0] + self.offset_x, pos[1] + self.offset_y)
    
    def stop_drag(self):
        self.dragging = False
    
    def reset_position(self):
        self.position = self.original_position