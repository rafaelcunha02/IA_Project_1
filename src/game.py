import pygame
import os
import random
from block import Block
from constants import *

class Game:
    def __init__(self, level):
        self.grid = self.load_level(level)
        self.level = level
        self.blocks = self.generate_blocks()
        self.score = 0
        self.game_over = False
        self.current_grid_pos = None
        self.can_place_current = False
    
    def load_level(self, level):
        grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        base_path = os.path.dirname(__file__)
        level_file = os.path.join(base_path, f'../levels/level{level}.txt')
        
        try:
            with open(level_file, 'r') as file:
                lines = file.readlines()
                for row, line in enumerate(lines):
                    for col, char in enumerate(line.strip()):
                        if char == 'R':
                            grid[row][col] = RED
                        elif char == 'G':
                            grid[row][col] = GREEN
                        elif char == 'B':
                            grid[row][col] = BLUE
        except FileNotFoundError:
            if(level != 0):
                print(f"Level file {level_file} not found. Loading empty grid.")
        
        return grid
    
    def generate_blocks(self):
        blocks = []
        
        # Positions for the three blocks at the bottom
        positions = [
            (50, WINDOW_HEIGHT - 120),
            (WINDOW_WIDTH // 2 - 40, WINDOW_HEIGHT - 120),
            (WINDOW_WIDTH - 180, WINDOW_HEIGHT - 120)
        ]
        
        for i in range(3):
            block_type = random.choice(BLOCK_TYPES)
            color = BLUE
            blocks.append(Block(block_type, color, positions[i]))
            
        return blocks
    
    def draw_grid(self):
        # Draw the grid background
        grid_rect = pygame.Rect(GRID_OFFSET_X, GRID_OFFSET_Y, 
                              GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        pygame.draw.rect(screen, (240, 240, 240), grid_rect)
        
        # Draw grid cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_rect = pygame.Rect(
                    GRID_OFFSET_X + col * CELL_SIZE,
                    GRID_OFFSET_Y + row * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                # Draw the filled cells if any
                if self.grid[row][col]:
                    pygame.draw.rect(screen, self.grid[row][col], cell_rect)
                
                # Draw cell borders
                pygame.draw.rect(screen, (200, 200, 200), cell_rect, 1)
    
    def draw_blocks(self):
        for block in self.blocks:
            block.draw()
    
    def draw_placement_preview(self, block):
        if not self.current_grid_pos:
            return
        
        grid_row, grid_col = self.current_grid_pos
        highlight_color = HIGHLIGHT_COLOR if self.can_place_current else INVALID_COLOR
        
        # Create a surface with per-pixel alpha
        highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(highlight_color)
        
        for row_idx, row in enumerate(block.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    r, c = grid_row + row_idx, grid_col + col_idx
                    
                    # Only draw preview if within grid bounds
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        pos_x = GRID_OFFSET_X + c * CELL_SIZE
                        pos_y = GRID_OFFSET_Y + r * CELL_SIZE
                        screen.blit(highlight_surface, (pos_x, pos_y))
    
    def draw_score(self):
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
    
    
    def draw_game_over(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        game_over_text = font.render("YOU LOST", True, (255, 0, 0))
        score_text = font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        menu_text = font.render("Press M to go to Menu", True, (255, 255, 255))
        
        screen.blit(game_over_text, (WINDOW_WIDTH // 2 - game_over_text.get_width() // 2, 
                                     WINDOW_HEIGHT // 2 - 60))
        screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, 
                                 WINDOW_HEIGHT // 2))
        screen.blit(menu_text, (WINDOW_WIDTH // 2 - menu_text.get_width() // 2, 
                                WINDOW_HEIGHT // 2 + 60))
    
    
    def can_place_block(self, block, grid_pos):
        if not grid_pos:
            return False
            
        grid_row, grid_col = grid_pos
        
        for row_idx, row in enumerate(block.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    r, c = grid_row + row_idx, grid_col + col_idx
                    
                    # Check boundaries
                    if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
                        return False
                    
                    # Check if cell is already occupied
                    if self.grid[r][c]:
                        return False
        
        return True
    
    def place_block(self, block, grid_pos):
        grid_row, grid_col = grid_pos
        
        for row_idx, row in enumerate(block.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    r, c = grid_row + row_idx, grid_col + col_idx
                    self.grid[r][c] = block.color
    
    def check_lines(self):
        lines_cleared = 0
        
        # Check horizontal lines
        rows_to_clear = []
        for row in range(GRID_SIZE):
            if all(self.grid[row]):
                rows_to_clear.append(row)
                lines_cleared += 1
        
        # Clear horizontal lines
        for row in rows_to_clear:
            self.grid[row] = [0 for _ in range(GRID_SIZE)]
        
        # Check vertical lines
        cols_to_clear = []
        for col in range(GRID_SIZE):
            if all(self.grid[row][col] for row in range(GRID_SIZE)):
                cols_to_clear.append(col)
                lines_cleared += 1
        
        # Clear vertical lines
        for col in cols_to_clear:
            for row in range(GRID_SIZE):
                self.grid[row][col] = 0
        
        # Update score
        self.score += lines_cleared * 100
        
        return lines_cleared > 0
    
    def find_grid_position(self, pixel_pos):
        x, y = pixel_pos
        
        # Check if position is within grid bounds
        if (GRID_OFFSET_X <= x <= GRID_OFFSET_X + GRID_SIZE * CELL_SIZE and
            GRID_OFFSET_Y <= y <= GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE):
            
            col = (x - GRID_OFFSET_X) // CELL_SIZE
            row = (y - GRID_OFFSET_Y) // CELL_SIZE
            
            return (row, col)
        
        return None
    
    def update_placement_preview(self, block):
        # Update the grid position based on the block's position
        self.current_grid_pos = self.find_grid_position(block.position)
        
        # Check if we can place the block at the current position
        self.can_place_current = self.can_place_block(block, self.current_grid_pos)
    
    def try_place_block(self, block):
        # Already checked in update_placement_preview
        if self.can_place_current and self.current_grid_pos:
            self.place_block(block, self.current_grid_pos)
            # Check if lines are cleared
            self.check_lines()
            # Check if game is over after placing the block
            #if self.check_game_over():
             #   self.game_over = True
            return True
        
        return False
    
    def check_game_over(self):
        # For each block, check if it can be placed anywhere on the grid
        for block in self.blocks:
            can_place_anywhere = False
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.can_place_block(block, (row, col)):
                        print(f"Block {block.shape} can be placed at ({row}, {col})")
                        can_place_anywhere = True
                        break
                if can_place_anywhere:
                    break
            
            if can_place_anywhere:
                return False
        
        return True
    
    def reset(self):
        self.grid = self.load_level(self.level)
        self.blocks = self.generate_blocks()
        self.score = 0
        self.game_over = False
        self.current_grid_pos = None
        self.can_place_current = False