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
        prev_score = game.score
        game.simulated_score = prev_score
        max_score = game.score
        max_aligned_reds = 0

        reds_changed = False
        score_changed = False
        #print("greedy move called with score: ", game.score)
        #print("reds: ", least_reds)
        #print("simulated score: ", game.simulated_score)
        #print("max score: ", max_score)

        for block in self.game.blocks:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    game.simulated_score = game.score
                    if self.game.can_place_block(block, (row, col)):
                        # Simulate placing the block
                        original_grid = [row[:] for row in self.game.grid]
                        self.game.current_grid_pos = (row, col)
                        self.game.simulate_try_place_block(block)
                        reds = self.game.count_reds()
                        # Undo the move
                        self.game.grid = original_grid

                        if reds < least_reds:
                            print("current vs previous least reds: ", reds, least_reds)
                            least_reds = reds
                            print("Greedy Move: ", block.shape, (row, col))
                            game.hint_block = block
                            game.hint_position = (row, col)
                            best_move = (block, (row, col))
                            reds_changed = True
                        elif not reds_changed and game.simulated_score > max_score:
                            print("current vs previous max score: ", game.simulated_score, max_score)
                            max_score = game.simulated_score
                            game.simulated_score = game.score
                            print("Greedy Move: ", block.shape, (row, col))
                            game.hint_block = block
                            game.hint_position = (row, col)
                            best_move = (block, (row, col))
                            score_changed = True
                        elif not reds_changed and not score_changed:
                            current_aligned_reds = game.count_aligned_reds(block.shape, (row, col))
                            if current_aligned_reds > max_aligned_reds:
                                max_aligned_reds = current_aligned_reds
                                game.hint_block = block
                                game.hint_position = (row, col)
                                best_move = (block, (row, col))
        
        game.current_grid_pos = None
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

