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
        least_reds = game.count_reds()
        prev_score = game.score
        game.simulated_score = prev_score
        max_score = game.score
        max_aligned_reds = 0
        max_aligned_blues = 0


        reds_changed = False
        score_changed = False
        can_align_reds = False
        can_align_blues = False


        #print("greedy move called with score: ", game.score)
        #print("reds: ", least_reds)
        #print("simulated score: ", game.simulated_score)
        #print("max score: ", max_score)
        for block in self.game.blocks:
            if (self.evaluate_grid() == 0):
                return (block,(0,0))
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    game.simulated_score = game.score
                    if self.game.can_place_block(block, (row, col)):
                        # Simulate placing the block
                        original_grid = [row[:] for row in self.game.grid]
                        self.game.current_grid_pos = (row, col)
                        self.game.simulate_try_place_block(block)
                        reds = self.game.count_reds()
                        #print("new reds:", reds)

                        #if self.game.grid == original_grid:
                        #    print("iguais!!!")
                        # Undo the move
                        self.game.grid = original_grid
                        if reds < least_reds:
                            #print("current vs previous least reds: ", reds, least_reds)
                            least_reds = reds
                            #print("Greedy Move: ", block.shape, (row, col))
                            game.hint_block = block
                            game.hint_position = (row, col)
                            best_move = (block, (row, col))
                            reds_changed = True
                            score_changed = False
                            can_align_reds = False
                            can_align_blues = False
                        elif not reds_changed and game.simulated_score > max_score:
                            #print("current vs previous max score: ", game.simulated_score, max_score)
                            max_score = game.simulated_score
                            game.simulated_score = game.score
                            #print("Greedy Move: ", block.shape, (row, col))
                            game.hint_block = block
                            game.hint_position = (row, col)
                            best_move = (block, (row, col))
                            score_changed = True
                            can_align_reds = False
                            can_align_blues = False
                        elif not reds_changed and not score_changed:
                            current_aligned_reds = game.count_aligned_reds(block.shape, (row, col))
                            #print("current_aligned_reds: ", current_aligned_reds)
                            if current_aligned_reds > max_aligned_reds:
                                can_align_reds = True
                                can_align_blues = False
                                max_aligned_reds = current_aligned_reds
                                game.hint_block = block
                                game.hint_position = (row, col)
                                best_move = (block, (row, col))
                            elif not can_align_reds:
                                can_align_blues = True
                                current_aligned_blues = game.count_aligned_blues(block.shape, (row, col))
                                if current_aligned_blues > max_aligned_blues:
                                    max_aligned_blues = current_aligned_blues
                                    game.hint_block = block
                                    game.hint_position = (row, col)
                                    best_move = (block, (row, col))
        
        game.current_grid_pos = None
        if(best_move):
            (b, (r, c)) = best_move
            #print("Best move being returned:", b.shape)
            #print("Row:", r)
            #print("Column:", c)
            reds_changed and print("DELETE REDS")
            score_changed and print("SCORE INCREASE")
            can_align_reds and print("ALIGN WITH REDS")
            can_align_blues and print("ALIGN WITH BLUES")
        else: 
            print("best move is none")
            print(can_align_reds)
            print("align reds?: ", can_align_reds)
            print("align blues?: ", can_align_blues)
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

