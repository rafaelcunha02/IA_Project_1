import random
from constants import *
from game import Game
from block import Block
from simulation import Simulation
import heapq
import itertools
import copy

class Bot:
    def __init__(self, game, mode):
        self.game = game
        self.mode = mode
        self.counter = itertools.count()
    
    def find_possible_moves(self, game):
        """Find all valid moves in the current game state.
        
        Returns:
            list: A list of tuples (block, position, simulation_data)
        """
        possible_moves = []
        
        for block in game.blocks:
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if not game.can_place_block(block, (row, col)):
                        continue
                    
                    original_grid = [row[:] for row in game.grid]
                    original_score = game.score
                    
                    game.current_grid_pos = (row, col)
                    game.simulate_try_place_block(block)
                    
                    simulation = Simulation(
                        reds=game.count_reds(),
                        score=game.simulated_score,
                        aligned_reds=game.count_aligned_reds(block.shape, (row, col)),
                        aligned_blues=game.count_aligned_blues(block.shape, (row, col)),
                    )
                    
                    game.grid = original_grid
                    game.simulated_score = original_score
                    

                    possible_moves.append((block, (row, col), simulation))
        
        game.current_grid_pos = None
        return possible_moves

    def evaluate_moves_greedy(self, game, possible_moves):
        """Evaluate moves using the greedy strategy.
        
        Args:
            game: The game instance
            possible_moves: List of (block, position, simulation_data) tuples
            
        Returns:
            tuple: The best move (block, position) according to greedy strategy
        """
        if not possible_moves:
            return None
            
        best_move = None
        least_reds = game.count_reds()
        max_score = game.score
        max_aligned_reds = 0
        max_aligned_blues = 0
        
        move_priority = {
            'reds_changed': False,
            'score_changed': False,
            'can_align_reds': False,
            'can_align_blues': False
        }
        
        for block, position, sim in possible_moves:
            if sim.reds < least_reds:
                least_reds = sim.reds
                best_move = (block, position)
                game.hint_block = block
                game.hint_position = position
                
                move_priority = {
                    'reds_changed': True,
                    'score_changed': False,
                    'can_align_reds': False,
                    'can_align_blues': False
                }
            
            elif not move_priority['reds_changed'] and sim.score > max_score:
                print("entrou score changed")
                max_score = sim.score
                best_move = (block, position)
                game.hint_block = block
                game.hint_position = position
                
                move_priority['score_changed'] = True
            
                move_priority['can_align_reds'] = False
                move_priority['can_align_blues'] = False
            
            elif not move_priority['reds_changed'] and not move_priority['score_changed']:
                if sim.aligned_reds > max_aligned_reds:
                    max_aligned_reds = sim.aligned_reds
                    best_move = (block, position)
                    game.hint_block = block
                    game.hint_position = position
                    
                    move_priority['can_align_reds'] = True
                    move_priority['can_align_blues'] = False
                    
                elif not move_priority['can_align_reds'] and sim.aligned_blues > max_aligned_blues:
                    max_aligned_blues = sim.aligned_blues
                    best_move = (block, position)
                    game.hint_block = block
                    game.hint_position = position
                    
                    move_priority['can_align_blues'] = True
        
        if best_move:
            move_priority['reds_changed'] and print("DELETE REDS")
            move_priority['score_changed'] and print("SCORE INCREASE")
            move_priority['can_align_reds'] and print("ALIGN WITH REDS")
            move_priority['can_align_blues'] and print("ALIGN WITH BLUES")
        else:
            print("best move is none")
            print("align reds?: ", move_priority['can_align_reds'])
            print("align blues?: ", move_priority['can_align_blues'])
        
        return best_move

    def auto_play_greedy(self, game):
        """Commander function that finds and evaluates moves using greedy strategy."""
        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(game)
        
        return self.evaluate_moves_greedy(game, possible_moves)
        

    def evaluate_grid(self):
        filled_cells = sum(cell != 0 for row in self.game.grid for cell in row)
        return filled_cells
    


    def heuristic(self, state):
        """Calculate heuristic value of state for A* search.
        
        A lower value indicates a better state.
        """
        # For level-based games with red blocks
        if self.game.level in [1, 2, 3]:
            # Prioritize states that remove red blocks
            return state.reds * 1000 - state.score
        else:
            # For endless mode, prioritize score
            return -state.score

    def simulate_move(self, game_state, block, position):
        """Simulate a move and return the resulting game state."""
        # Create a deep copy of the game state to avoid modifying the original
        new_game = copy.deepcopy(game_state)
        
        # Set the current grid position and try to place the block
        new_game.current_grid_pos = position
        new_game.try_place_block(block)
        
        return new_game
    
    