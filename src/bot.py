import random
from constants import *
from game import Game
from block import Block

class Bot:
    def __init__(self, game, mode):
        self.game = game
        self.mode = mode

    def get_greedy_move(self, game):
        best_move = None
        least_reds = game.count_reds()

        for block in self.game.blocks:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.game.can_place_block(block, (row, col)):
                        # Simulate placing the block
                        original_grid = [row[:] for row in self.game.grid]
                        self.game.current_grid_pos = (row, col)
                        self.game.try_place_block(block)
                        reds = self.game.count_reds()
                        # Undo the move
                        self.game.grid = original_grid

                        if reds < least_reds:
                            least_reds = reds
                            print("Greedy Move: ", block.shape, (row, col))
                            game.hint_block = block
                            game.hint_position = (row, col)
                            best_move = (block, (row, col))

        return best_move

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

