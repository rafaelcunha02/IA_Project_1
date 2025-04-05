import random
from constants import *
from game import Game
from block import Block
from simulation import Simulation
import heapq
import itertools
import copy
import os
import csv

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
                    if not game.can_place_block(block, (row, col), game.grid_size):
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
    

    def simulate_move(self, game_state, block, position):
        """Simulate a move and return the resulting game state."""
        # Create a deep copy of the game state to avoid modifying the original
        new_game = copy.deepcopy(game_state)  # Use deepcopy to ensure the original game state is not altered
    
        # Set the current grid position and try to place the block
        new_game.current_grid_pos = position
        new_game.try_place_block(block)
        
        for i, bloco in enumerate(new_game.blocks):
            if bloco.shape == block.shape:
                del new_game.blocks[i]
                
                if not new_game.blocks:
                    new_game.blocks = new_game.generate_blocks()
                
                #print("Removed block with shape:", bloco.shape)
                break
    
        # Create a new Simulation object based on the modified game state
        new_game_sim = Simulation(
            reds=new_game.count_reds(),
            score=new_game.score,
            aligned_reds=new_game.count_aligned_reds(block.shape, position),
            aligned_blues=new_game.count_aligned_blues(block.shape, position),
            game=new_game,  # Pass the modified game state
            valid_moves=self.find_possible_moves(new_game)  # Find valid moves for the new state
        )
    
        return new_game_sim
    


    def greedy_bestfs_algorithm(self, current_state, goal_state, possible_moves):
        # Initialize current_state's valid_moves if not set
        if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
            current_state.valid_moves = possible_moves
            
        open_set = []
        closed_set = set()
        g_score = {current_state: 0}
        f_score = {current_state: self.heuristic(current_state, goal_state)}
        parents = {current_state: (None, None, 0)}  # Track parent relationships and depth

        open_set.append(current_state)

        # Track the closest score to the goal
        closest_f_score = f_score[current_state]

        while open_set:
            current = min(open_set, key=lambda state: f_score.get(state, float('inf')))

            # Check if goal is reached
            print(current.reds)
            if current.reds == 0:
                print("goal state")
                return self.reconstruct_move(parents, current)

            open_set.remove(current)
            closed_set.add(current)

            # Update the closest state if necessary
            if f_score[current] < closest_f_score:
                closest_f_score = f_score[current]

            # Get the current depth
            _, _, current_depth = parents[current]

            
            # Use the valid_moves from the CURRENT state, not the initial state
            if not hasattr(current, 'valid_moves') or current.valid_moves is None:
                # If this state doesn't have valid_moves calculated, do it now
                current.valid_moves = self.find_possible_moves(current.game)
            print(f"Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")

                # Or more detailed tracking inside the move loop:
            for move in current.valid_moves:
                 # Current code...
                (block, (row, col), simulation) = move
                neighbor = self.simulate_move(current.game, block, (row, col))
                # print(f"Considering move: {move}, neighbor reds: {neighbor.reds}")

                tentative_g_score = g_score[current]
                
                # If this path to neighbor is worse than one we've already found, skip
                if tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                    
                # We found a better path to the neighbor
                parents[neighbor] = (current, move, current_depth + 1)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_state)
                
                if neighbor in closed_set:
                    # Important: move the neighbor from closed back to open set
                    # since we found a better path to it
                    closed_set.remove(neighbor)
                    
                if neighbor not in open_set:
                    open_set.append(neighbor)

        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found.")
        return None
    
    def is_goal_state(self, state, goal_state):
        # Custom logic to check if the goal is reached
        return state.reds == goal_state.reds

    def heuristic(self, state, goal_state):
        # If there are no valid moves, return a very high score to deprioritize this state

        if state.game.game_over:
            return float('inf')  # Infinite score for states with no valid moves

        if(goal_state.reds != 0):
            return(
                - state.score
                - state.aligned_blues * 10
            )
        # Otherwise, calculate the heuristic based on the priorities
        return (
            state.reds * 1000 -  # Prioritize fewer reds
            state.score * 2 -    # Then prioritize more score
            state.aligned_reds * 100 -  # Then prioritize more aligned reds
            state.aligned_blues * 10  # Then prioritize more aligned blues
        )
    
    def heuristic_astar(self, state):
        # Count the number of red squares remaining
        red_squares = state.reds

        # Count the number of rows and columns with red squares
        rows_with_reds = set()
        cols_with_reds = set()
        for row in range(len(state.game.grid)):
            for col in range(len(state.game.grid[row])):
                if state.game.grid[row][col] == (255, 0, 0):  # Red square
                    rows_with_reds.add(row)
                    cols_with_reds.add(col)

        # Estimate the minimum number of moves required to clear all rows and columns with red squares
        rows_to_clear = len(rows_with_reds)
        cols_to_clear = len(cols_with_reds)

        # Assume that each move can clear at most one row or column
        moves_to_clear_reds = max(rows_to_clear, cols_to_clear)

        # Final heuristic value
        return moves_to_clear_reds

    def reconstruct_move(self, parents, current):
        # Trace back to the first move
        moves = []  # List to store the sequence of moves
        while current in parents and parents[current][0] is not None:  # While there is a parent
            parent, move, _ = parents[current]  # Extract parent, move, and depth
            if move:  # If a move exists, add it to the list
                moves.append(move)
            current = parent  # Move to the parent state
    
        # Reverse the moves to get the sequence from start to goal
        moves.reverse()
    
        # Print the sequence of moves
        print("Sequence of moves:")
        for i, move in enumerate(moves):
            block, position, _ = move
            print(f"Move {i + 1}: Block shape: {block.shape}, Position: {position}")
        # Return the first move in the sequence (if it exists)
        return moves

    def cost(self, current, move):
        # Implement logic to calculate the cost of a move
        return 1  # Example cost for each move
    
    def auto_play_greedy_bestfs(self, auto = True):
        
        """Commander function that finds and evaluates moves using A* algorithm."""
        if auto:
            if(self.game.solution != []):
                self.game.solution.pop(0)
                print(self.game.solution[0])
                return self.game.solution

        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(self.game)
    
        current_state = Simulation(
            reds=self.game.count_reds(),
            score=self.game.score,
            aligned_reds=0,
            aligned_blues=0,
            game=self.game
        )
        
        if(self.game.level > 3 or self.game.level < 1):
            print("LEVEL")
            print(self.game.level)
            goal_state = Simulation(None, float('inf'), None, None, None)
        else:
            goal_state = Simulation(0, None, None, None, None)
        
        a = self.greedy_bestfs_algorithm(current_state, goal_state, possible_moves)
        self.game.solution = a
        (block, (row, col), sim) = a[0]
        self.game.hint_block = block
        self.game.hint_position = (row, col)
        if auto:
            return a
        else:
            return a[0]


    def bfs_algorithm(self, initial_state, goal_state, possible_moves):
        """Perform a BFS to find the optimal move sequence."""
        from collections import deque
        
        # Initialize the BFS queue with the initial state
        queue = deque([(initial_state, None, None, 0)])  # (state, parent, move, depth)
        visited = set()
        parents = {}  # Track parent relationships for path reconstruction
    
        
        print("lel")
        counter = 0
        while queue:
            current_state, parent, move_taken, depth = queue.popleft()
            # Store the parent relationship
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
            
            # Check if the current state is the goal state
            if current_state.reds == 0:
                print("goal state found")
                return self.reconstruct_move(parents, current_state)
        
            
            # Create a state key for visited tracking
            state_key = str(current_state.game.grid) + str(current_state.reds) + str(current_state.score)
            if state_key in visited:
                continue
            counter += 1
            visited.add(state_key)
            
            # Generate all possible moves from the current state
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
            
            for move in current_state.valid_moves:
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
                
                # Add the next state to the queue
                queue.append((next_state, current_state, move, depth + 1))
        
        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        print(counter)
        return None
    

    def dfs_algorithm(self, initial_state, goal_state, possible_moves):
        # Initialize the DFS stack with the initial state
        stack = [(initial_state, None, None, 0)]  # (state, parent, move, depth)
        visited = set()
        parents = {}  # Track parent relationships for path reconstruction
        
        # print(f"Initial stack: {stack}")
        counter = 0
        while stack:
            current_state, parent, move_taken, depth = stack.pop()  

            # Store the parent relationship
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
            
            # Check if the current state is the goal state
            if self.is_goal_state(current_state, goal_state):
                return self.reconstruct_move(parents, current_state)
            
            # Create a state key for visited tracking
            state_key = hash(str(current_state.game.grid) + str(current_state.reds) + str(current_state.score))
            if state_key in visited:
                continue
            counter += 1
            visited.add(state_key)
            
            
            # Generate all possible moves from the current state
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
                print(f"Valid moves for current state: {current_state.valid_moves}")
            
            # For DFS, we reverse the order to explore depth-first
            for move in reversed(current_state.valid_moves):
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
                
                # Add the next state to the stack
                stack.append((next_state, current_state, move, depth + 1))
        
        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        print(counter)
        return None
    
    def iterative_deepning_algorithm(self, initial_state, goal_state, possible_moves, depth_limit, goal_state_reached):

        # Initialize the DFS stack with the initial state
        stack = [(initial_state, None, None, 0)]  # (state, parent, move, depth)
        visited = set()
        parents = {}  # Track parent relationships for path reconstruction
        
        # Track the closest state to the goal
        closest_state = initial_state
        closest_score = self.heuristic(initial_state, goal_state)
        
        # print(f"Initial stack: {stack}")
        counter = 0
        while stack:
            current_state, parent, move_taken, depth = stack.pop()  

            # Store the parent relationship
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
            
            # Check if the current state is the goal state
            if self.is_goal_state(current_state, goal_state):
                print("goal state")
                goal_state_reached[0] = True
                return self.reconstruct_move(parents, current_state)
            
            # Stop expanding if the depth limit is reached
            if depth >= depth_limit:
                continue
            
            # Create a state key for visited tracking
            state_key = hash(str(current_state.game.grid) + str(current_state.reds) + str(current_state.score))
            if state_key in visited:
                continue
            counter += 1
            visited.add(state_key)
            
            # Update the closest state if necessary
            current_score = self.heuristic(current_state, goal_state)
            if current_score < closest_score:
                closest_state = current_state
                closest_score = current_score
            
            # Generate all possible moves from the current state
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
                print(f"Valid moves for current state: {current_state.valid_moves}")
            
            # For DFS, we reverse the order to explore depth-first
            for move in reversed(current_state.valid_moves):
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
                
                # Add the next state to the stack
                stack.append((next_state, current_state, move, depth + 1))
        
        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        print(counter)
        return self.reconstruct_move(parents, closest_state)


    def auto_play_bfs(self):
        """Commander function that finds and evaluates moves using BFS."""
        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(self.game)
        
        initial_state = Simulation(
            reds=self.game.count_reds(),
            score=self.game.score,
            aligned_reds=0,
            aligned_blues=0,
            game=self.game,
            valid_moves=possible_moves  # Pass the valid moves to the initial state
        )
        

        if(self.game.level > 4 or self.game.level < 1):
            print("LEVEL")
            print(self.game.level)
            goal_state = Simulation(None, float('inf'), None, None, None)
        else:
            goal_state = Simulation(0, None, None, None, None)
        
        self.game.solution = self.bfs_algorithm(initial_state, goal_state, possible_moves)
        return self.game.solution
    

    def auto_play_dfs(self):
        """Commander function that finds and evaluates moves using DFS."""
        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(self.game)
        
        initial_state = Simulation(
            reds=self.game.count_reds(),
            score=self.game.score,
            aligned_reds=0,
            aligned_blues=0,
            game=self.game,
            valid_moves=possible_moves  # Pass the valid moves to the initial state
        )
        
        if(self.game.level > 4 or self.game.level < 1):
            print("LEVEL")
            print(self.game.level)
            goal_state = Simulation(None, float('inf'), None, None, None)
        else:
            goal_state = Simulation(0, None, None, None, None)
        
        self.game.solution = self.dfs_algorithm(initial_state, goal_state, possible_moves)
        return self.game.solution
    
    def auto_play_iterative_deepning(self):
        """Commander function that finds and evaluates moves using DFS."""
        if self.game.solution != []:
            self.game.solution.pop(0)
            print(self.game.solution[0])
            return self.game.solution
        
        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(self.game)
        
        initial_state = Simulation(
            reds=self.game.count_reds(),
            score=self.game.score,
            aligned_reds=0,
            aligned_blues=0,
            game=self.game,
            valid_moves=possible_moves  # Pass the valid moves to the initial state
        )
        
        if(self.game.level > 4 or self.game.level < 1):
            print("LEVEL")
            print(self.game.level)
            goal_state = Simulation(None, float('inf'), None, None, None)
        else:
            goal_state = Simulation(0, None, None, None, None)
        
        for i in range(1, 9999999, 1):
                goal_state_reached = [False]
                a = self.iterative_deepning_algorithm(initial_state, goal_state, possible_moves, i, goal_state_reached)
                if(goal_state_reached[0] == True):
                    self.game.solution = a
                    return a
    
    def astar_algorithm(self, current_state, goal_state, possible_moves, depth_limit):
        # Initialize current_state's valid_moves if not set
        if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
            current_state.valid_moves = possible_moves
            
        open_set = []
        closed_set = set()
        g_score = {current_state: 0}
        f_score = {current_state: self.heuristic_astar(current_state)}
        parents = {current_state: (None, None, 0)}  # Track parent relationships and depth

        open_set.append(current_state)

        # Track the closest state to the goal
        closest_state = current_state
        closest_f_score = f_score[current_state]

        while open_set:
            current = min(open_set, key=lambda state: f_score.get(state, float('inf')))

            # Check if goal is reached
            print(current.reds)
            if current.reds == 0:
                print("goal state")
                return self.reconstruct_move(parents, current)

            open_set.remove(current)
            closed_set.add(current)

            # Update the closest state if necessary
            if f_score[current] < closest_f_score:
                closest_state = current
                closest_f_score = f_score[current]

            # Get the current depth
            _, _, current_depth = parents[current]

            # Stop expanding if the depth limit is reached
            if current_depth >= depth_limit:
                continue
            
            # Use the valid_moves from the CURRENT state, not the initial state
            if not hasattr(current, 'valid_moves') or current.valid_moves is None:
                # If this state doesn't have valid_moves calculated, do it now
                current.valid_moves = self.find_possible_moves(current.game)
            print(f"Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")

                # Or more detailed tracking inside the move loop:
            for move in current.valid_moves:
                 # Current code...
                (block, (row, col), simulation) = move
                neighbor = self.simulate_move(current.game, block, (row, col))
                # print(f"Considering move: {move}, neighbor reds: {neighbor.reds}")

                tentative_g_score = g_score[current] + self.cost(current, move)
                
                # If this path to neighbor is worse than one we've already found, skip
                if tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                    
                # We found a better path to the neighbor
                parents[neighbor] = (current, move, current_depth + 1)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic_astar(neighbor)
                
                if neighbor in closed_set:
                    # Important: move the neighbor from closed back to open set
                    # since we found a better path to it
                    closed_set.remove(neighbor)
                    
                if neighbor not in open_set:
                    open_set.append(neighbor)

        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        return self.reconstruct_move(parents, closest_state)
    
    def auto_play_astar(self):
        """Commander function that finds and evaluates moves using A* algorithm."""
        if(self.game.solution != []):
            self.game.solution.pop(0)
            print(self.game.solution[0])
            return self.game.solution
        
        if self.evaluate_grid() == 0:
            return (self.game.blocks[0], (0, 0))
        
        possible_moves = self.find_possible_moves(self.game)
    
        current_state = Simulation(
            reds=self.game.count_reds(),
            score=self.game.score,
            aligned_reds=0,
            aligned_blues=0,
            game=self.game
        )
        
        if(self.game.level > 3 or self.game.level < 1):
            print("LEVEL")
            print(self.game.level)
            goal_state = Simulation(None, float('inf'), None, None, None)
        else:
            goal_state = Simulation(0, None, None, None, None)
        
        # depth_limit = len(self.game.blocks)  # Limit depth to the number of blocks
        depth_limit = 10
        self.game.solution = self.astar_algorithm(current_state, goal_state, possible_moves, depth_limit)
        return self.game.solution
    

    def auto_play(self):
        """Commander function that finds and evaluates moves using the selected mode."""
        if self.game.player_type == 1:            
            return self.auto_play_greedy_bestfs()
        elif self.game.player_type == 2:
            return self.auto_play_bfs()
        elif self.game.player_type == 3:
            return self.auto_play_dfs()
        elif self.game.player_type == 4:
            return self.auto_play_iterative_deepning()
        elif self.game.player_type == 5:
            return self.auto_play_astar()
        else:
            print("Invalid mode. Please select 'greedy', 'astar', or 'bfs'.")
            return None