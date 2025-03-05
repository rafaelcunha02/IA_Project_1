import random
from constants import *
from game import Game
from block import Block

class Bot:
    def __init__(self, game, mode):
        self.game = game
        self.mode = mode

    def auto_play_greedy(self, game):
        best_move = None

        for block in self.game.blocks:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    # Copy grid
                    simulated_grid = [row[:] for row in self.game.grid]
                    
                    if self.game.can_place_block(block, (row, col)):
                        # Temporary Game State
                        original_grid = self.game.grid
                        self.game.grid = simulated_grid
                        
                        try:
                            # Simulate block placement / new red amount / score calculation
                            self.game.simulate_try_place_block(block)
                            reds = self.game.count_reds()
                            simulated_score = self.game.score
                            
                            # Priority:
                            # 1. Reduce reds
                            # 2. Increase score
                            # 3. Maximize aligned reds
                            move_priority = (
                                -reds,  # Lower reds is better
                                simulated_score,
                                game.count_aligned_reds(block.shape, (row, col))
                            ) # (When we compare tuples in Cobrinhas, they're compared using the first element, remember FPRO)
                            
                            # Update best move better move found
                            if best_move is None or move_priority > best_move[1]: # if current move priority is better than best move's move priority 
                                best_move = (
                                    (block, (row, col)),  # Move
                                    move_priority  # Priority score for comparison
                                )
                                game.hint_position = (row, col)
                                game.hint_block = block # Hint case for human mode
                        
                        finally:
                            # reset grid original grid
                            self.game.grid = original_grid
        
        # Return move only, not move with priority score associated
        return best_move[0] if best_move else None

    def evaluate_grid(self):
        # Simple evaluation function: count the number of filled cells
        filled_cells = sum(cell != 0 for row in self.game.grid for cell in row)
        return filled_cells

# Usage
# game = Game(level)
# bot = Bot(game)
# best_move = bot.get_best_move()
# if best_move:
#     block, position = best_move
#     game.place_block(block, position)

