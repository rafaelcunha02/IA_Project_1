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
    

    def simulate_move(self, game_state, block, position):
        """Simulate a move and return the resulting game state."""
        # Create a deep copy of the game state to avoid modifying the original
        new_game = copy.deepcopy(game_state)  # Use deepcopy to ensure the original game state is not altered
    
        # Set the current grid position and try to place the block
        new_game.current_grid_pos = position
        new_game.try_place_block(block)
    
        #print("Attempting to remove block with shape:", block.shape)
        #print("Blocks before removal:", [b.shape for b in new_game.blocks])
        
        for i, bloco in enumerate(new_game.blocks):
            if bloco.shape == block.shape:
                del new_game.blocks[i]
                #print("Removed block with shape:", bloco.shape)
                break
        
        #print("Blocks after removal:", [b.shape for b in new_game.blocks])
    
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
    


    def a_star_algorithm(self, current_state, goal_state, possible_moves, depth_limit):
        # Initialize current_state's valid_moves if not set
        if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
            current_state.valid_moves = possible_moves
            
        open_set = []
        closed_set = set()
        g_score = {current_state: 0}
        f_score = {current_state: self.heuristic(current_state, goal_state)}
        parents = {current_state: (None, None, 0)}  # Track parent relationships and depth

        open_set.append(current_state)

        # Track the closest state to the goal
        closest_state = current_state
        closest_f_score = f_score[current_state]

        while open_set:
            current = min(open_set, key=lambda state: f_score.get(state, float('inf')))

            # Check if goal is reached
            if self.is_goal_state(current, goal_state):
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
                
            for move in current.valid_moves:  # Use current.valid_moves here, not current_state.valid_moves
                (block, (row, col), simulation) = move
                neighbor = self.simulate_move(current.game, block, (row, col))

                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + self.cost(current, move)

                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                # Update scores, parent, and depth
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_state)
                parents[neighbor] = (current, move, current_depth + 1)  # Increment depth

        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        return self.reconstruct_move(parents, closest_state)
    
    def is_goal_state(self, state, goal_state):
        # Custom logic to check if the goal is reached
        return state.reds == goal_state.reds

    def heuristic(self, state, goal_state):
        return (
            state.reds * 1000 -  # Prioritize fewer reds
            state.score * 2 -
            state.aligned_reds * 10 -  # Prioritize more aligned reds
            state.aligned_blues * 10  # Prioritize more aligned blues
        )


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
    
        if(moves[0]):
            (block, (row, col), simulation) = moves[0]
        # Return the first move in the sequence (if it exists)
        return (block, (row,col)) if moves else None

    def cost(self, current, move):
        # Implement logic to calculate the cost of a move
        return 1  # Example cost for each move
    
    def auto_play_astar(self):
        """Commander function that finds and evaluates moves using A* algorithm."""
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
        
        goal_state = Simulation(0, None, None, None, None)
        
        depth_limit = len(self.game.blocks)  # Limit depth to the number of blocks
        return self.a_star_algorithm(current_state, goal_state, possible_moves, depth_limit)




    def bfs_algorithm(self, initial_state, goal_state, possible_moves, depth_limit):
        """Perform a BFS to find the optimal move sequence."""
        from collections import deque
        
        # Initialize the BFS queue with the initial state
        queue = deque([(initial_state, None, None, 0)])  # (state, parent, move, depth)
        visited = set()
        parents = {}  # Track parent relationships for path reconstruction
        
        # Track the closest state to the goal
        closest_state = initial_state
        closest_score = self.heuristic(initial_state, goal_state)
        
        while queue:
            current_state, parent, move_taken, depth = queue.popleft()
            
            # Store the parent relationship
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
            
            # Check if the current state is the goal state
            if self.is_goal_state(current_state, goal_state):
                return self.reconstruct_move(parents, current_state)
            
            # Stop expanding if the depth limit is reached
            if depth >= depth_limit:
                continue
            
            # Create a state key for visited tracking
            state_key = hash(str(current_state.game.grid) + str(current_state.reds) + str(current_state.score))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Update the closest state if necessary
            current_score = self.heuristic(current_state, goal_state)
            if current_score < closest_score:
                closest_state = current_state
                closest_score = current_score
            
            # Generate all possible moves from the current state
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
            
            for move in current_state.valid_moves:
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
                
                # Add the next state to the queue
                queue.append((next_state, current_state, move, depth))
        
        # If no goal state is found, return the move leading to the closest state
        print("Goal state not found. Returning closest state.")
        return self.reconstruct_move(parents, closest_state)
    

    def dfs_algorithm(self, initial_state, goal_state, possible_moves, depth_limit):
        # Initialize the DFS stack with the initial state
        stack = [(initial_state, None, None, 0)]  # (state, parent, move, depth)
        visited = set()
        parents = {}  # Track parent relationships for path reconstruction
        
        # Track the closest state to the goal
        closest_state = initial_state
        closest_score = self.heuristic(initial_state, goal_state)
        
        # print(f"Initial stack: {stack}")
        while stack:
            current_state, parent, move_taken, depth = stack.pop()  

            # Store the parent relationship
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
            
            # Check if the current state is the goal state
            if self.is_goal_state(current_state, goal_state):
                return self.reconstruct_move(parents, current_state)
            
            # Stop expanding if the depth limit is reached
            if depth >= depth_limit:
                continue
            
            # Create a state key for visited tracking
            state_key = hash(str(current_state.game.grid) + str(current_state.reds) + str(current_state.score))
            if state_key in visited:
                continue
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
        
        goal_state = Simulation(0, None, None, None, None, None)
        
        depth_limit = len(self.game.blocks) + 1  # Limit depth to the number of blocks
        return self.bfs_algorithm(initial_state, goal_state, possible_moves, depth_limit)
    

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
        
        goal_state = Simulation(0, None, None, None, None, None)
        
        depth_limit = len(self.game.blocks) + 1  # Limit depth to the number of blocks
        return self.dfs_algorithm(initial_state, goal_state, possible_moves, depth_limit)
    

    def auto_play(self):
        """Commander function that finds and evaluates moves using the selected mode."""
        if self.game.player_type == 1:
            return self.auto_play_greedy(self.game)
        elif self.game.player_type == 2:
            return self.auto_play_bfs()
        elif self.game.player_type == 3:
            return self.auto_play_dfs()
        elif self.game.player_type == 4:
            return self.auto_play_astar()
        else:
            print("Invalid mode. Please select 'greedy', 'astar', or 'bfs'.")
            return None