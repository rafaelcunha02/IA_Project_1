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
import time
from collections import deque
import sys

def write_to_csv(filename, data, headers=["Number of Moves", "Number of States", "Memory", "Time"]):
    file_exists = False
    try:
        file_exists = open(filename).close() is None
    except FileNotFoundError:
        pass

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        if headers and not file_exists:
            writer.writerow(headers)

        writer.writerows(data)


class Bot:
    def __init__(self, game, mode):
        self.game = game
        self.mode = mode
        self.counter = itertools.count()
    
    def find_possible_moves(self, game):
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

    def evaluate_grid(self):
        filled_cells = sum(cell != 0 for row in self.game.grid for cell in row)
        return filled_cells
    

    def simulate_move(self, game_state, block, position):
        new_game = copy.deepcopy(game_state)  
        new_game.current_grid_pos = position
        new_game.try_place_block(block)
        
        for i, bloco in enumerate(new_game.blocks):
            if bloco.shape == block.shape:
                del new_game.blocks[i]
                
                if not new_game.blocks:
                    new_game.blocks = new_game.generate_blocks()
                
                break

        new_game_sim = Simulation(
            reds=new_game.count_reds(),
            score=new_game.score,
            aligned_reds=new_game.count_aligned_reds(block.shape, position),
            aligned_blues=new_game.count_aligned_blues(block.shape, position),
            game=new_game,  
            valid_moves=self.find_possible_moves(new_game),  
            special=self.game.player_type == 6
        )
    
        return new_game_sim
    

    def greedy_bestfs_algorithm(self, current_state, goal_state, possible_moves):
        start_time = time.time()
        counter = 0
    
        if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
            current_state.valid_moves = possible_moves
    
        open_set = []
        closed_set = set()
        g_score = {current_state: 0}
        f_score = {current_state: self.heuristic(current_state, goal_state)}
        parents = {current_state: (None, None, 0)} 
    
        heapq.heappush(open_set, (f_score[current_state], current_state))
    
        closest_f_score = f_score[current_state]
    
        while open_set:
            _, current = heapq.heappop(open_set)
    
            if current.reds == 0:
                elapsed_time = time.time() - start_time
                moves = self.reconstruct_move(parents, current)
                num_moves = len(moves)
                memory_used = sys.getsizeof(open_set) + sys.getsizeof(closed_set) + sys.getsizeof(parents)

                write_to_csv(
                    f"greedy_algorithm_metrics_{self.game.level}.csv",
                    [[num_moves, counter, memory_used, elapsed_time]],
                    headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                )

                print("Goal state reached")
                print(f"Number of states visited: {counter}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Memory used: {memory_used} bytes")
                return self.reconstruct_move(parents, current)
    
            closed_set.add(current)
            counter += 1


            if f_score[current] < closest_f_score:
                closest_f_score = f_score[current]
    
            _, _, current_depth = parents[current]
    
            if not hasattr(current, 'valid_moves') or current.valid_moves is None:
                current.valid_moves = self.find_possible_moves(current.game)
    
            for move in current.valid_moves:
                (block, (row, col), simulation) = move
                neighbor = self.simulate_move(current.game, block, (row, col))
    
                tentative_g_score = g_score[current]
    
                if tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
    
                parents[neighbor] = (current, move, current_depth + 1)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor,goal_state)
    
                if neighbor in closed_set:
                    continue
    
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                print(f"Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")

        print("Goal state not found")
        return None
    
    
    def is_goal_state(self, state, goal_state):
        return state.reds == goal_state.reds

    def heuristic(self, state, goal_state):
        if state.game.game_over:
            return float('inf') 

        if(goal_state.reds != 0):
            return(
                - state.score
                - state.aligned_blues * 10
            )

        return (
            state.reds * 1000 -  
            state.score * 2 -    
            state.aligned_reds * 100 -  
            state.aligned_blues * 10  
        )
    
    def heuristic_astar(self, state):
        red_squares = state.reds
        rows_with_reds = set()
        cols_with_reds = set()
        for row in range(len(state.game.grid)):
            for col in range(len(state.game.grid[row])):
                if state.game.grid[row][col] == (255, 0, 0): 
                    rows_with_reds.add(row)
                    cols_with_reds.add(col)

        rows_to_clear = len(rows_with_reds)
        cols_to_clear = len(cols_with_reds)

        moves_to_clear_reds = max(rows_to_clear, cols_to_clear)
        isolated_reds = red_squares - (rows_to_clear + cols_to_clear)
        penalty_for_isolated_reds = max(0, isolated_reds // 2)

        return moves_to_clear_reds 

    def reconstruct_move(self, parents, current):
        moves = []  
        while current in parents and parents[current][0] is not None:  
            parent, move, _ = parents[current]  
            if move:  
                (block, position, _) = move
                print("block shape:", block.shape)
                print("position:", position)
                print("\n")
                moves.append(move)
            current = parent 
    
        moves.reverse()
        print("Sequence of moves:")

        for i, move in enumerate(moves):
            block, position, _ = move
            print(f"Move {i + 1}: Block shape: {block.shape}, Position: {position}")

        return moves

    def cost(self, current, move):
        return 1  
    
    def auto_play_greedy_bestfs(self, auto = True):
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
        start_time = time.time()
        queue = deque([(initial_state, None, None, 0)])  
        visited = set() 
        parents = {}
    
        counter = 0
        while queue:
            current_state, parent, move_taken, depth = queue.popleft()
    
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
    
            if current_state.reds == 0:
                elapsed_time = time.time() - start_time
                moves = self.reconstruct_move(parents, current_state)
                num_moves = len(moves)
                memory_used = sys.getsizeof(queue) + sys.getsizeof(visited) + sys.getsizeof(parents)
    
                write_to_csv(
                    f"bfs_algorithm_metrics_{self.game.level}.csv",
                    [[num_moves, counter, memory_used, elapsed_time]],
                    headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                )
    
                print("Goal state reached")
                print(f"Number of states visited: {counter}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Memory used: {memory_used} bytes")
                return moves
    
            if current_state in visited:
                continue
            visited.add(current_state)  
            counter += 1
    
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
    
            for move in current_state.valid_moves:
                block, position, sim = move
                next_state = self.simulate_move(current_state.game, block, position)
    
                queue.append((next_state, current_state, move, depth + 1))
    
        print("Goal state not found")
        print(f"Number of states visited: {counter}")
        return None    


    def dfs_algorithm(self, initial_state, goal_state, possible_moves):
        stack = [(initial_state, None, None, 0)]  
        visited = set()  
        parents = {}  
    
        start_time = time.time()
        counter = 0
    
        while stack:
            current_state, parent, move_taken, depth = stack.pop()
    
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)
    
            if self.is_goal_state(current_state, goal_state):
                elapsed_time = time.time() - start_time
                moves = self.reconstruct_move(parents, current_state)
                num_moves = len(moves)
                memory_used = sys.getsizeof(stack) + sys.getsizeof(visited) + sys.getsizeof(parents)
    
                write_to_csv(
                    f"dfs_algorithm_metrics_{self.game.level}.csv",
                    [[num_moves, counter, memory_used, elapsed_time]],
                    headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                )
    
                print("Goal state reached")
                print(f"Number of states visited: {counter}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Memory used: {memory_used} bytes")
                return moves
    
            if current_state in visited:
                continue
            visited.add(current_state) 
            counter += 1
    
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
    
            for move in reversed(current_state.valid_moves):
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
    
                stack.append((next_state, current_state, move, depth + 1))
    
        print("Goal state not found")
        print(f"Number of states visited: {counter}")
        return None
    
    
    def iterative_deepning_algorithm(self, initial_state, goal_state, possible_moves, depth_limit, goal_state_reached):
        stack = [(initial_state, None, None, 0)] 
        visited = set()  
        parents = {}  
    
        closest_state = initial_state
        closest_score = self.heuristic(initial_state, goal_state)
    
        start_time = time.time()
        counter = 0
    
        while stack:
            current_state, parent, move_taken, depth = stack.pop()
    
            if parent is not None:
                parents[current_state] = (parent, move_taken, depth)

            if self.is_goal_state(current_state, goal_state):
                elapsed_time = time.time() - start_time
                moves = self.reconstruct_move(parents, current_state)
                num_moves = len(moves)
                memory_used = sys.getsizeof(stack) + sys.getsizeof(visited) + sys.getsizeof(parents)
    
                write_to_csv(
                    f"iterative_deepening_metrics_{self.game.level}.csv",
                    [[num_moves, counter, memory_used, elapsed_time]],
                    headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                )
    
                print("Goal state reached")
                print(f"Number of states visited: {counter}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Memory used: {memory_used} bytes")
                goal_state_reached[0] = True
                return moves, elapsed_time, memory_used, counter
    
            if depth >= depth_limit:
                continue
    
            if current_state in visited:
                continue
            visited.add(current_state) 
            counter += 1
    
            current_score = self.heuristic(current_state, goal_state)
            if current_score < closest_score:
                closest_state = current_state
                closest_score = current_score
    
            if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
                current_state.valid_moves = self.find_possible_moves(current_state.game)
    
            for move in reversed(current_state.valid_moves):
                block, position, _ = move
                next_state = self.simulate_move(current_state.game, block, position)
    
                stack.append((next_state, current_state, move, depth + 1))
    
        print("Goal state not found. Returning closest state.")
        elapsed_time = time.time() - start_time
        memory_used = sys.getsizeof(stack) + sys.getsizeof(visited) + sys.getsizeof(parents)
    
        write_to_csv(
            f"iterative_deepening_metrics_{self.game.level}.csv",
            [[0, counter, memory_used, elapsed_time]],
            headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
        )
    
        print(f"Number of states visited: {counter}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Memory used: {memory_used} bytes")
        return elapsed_time, memory_used, counter


    def auto_play_bfs(self):
        """Commander function that finds and evaluates moves using BFS."""

        if(self.game.solution != []):
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
            valid_moves=possible_moves  
        )
        
        goal_state = Simulation(0, None, None, None, None)
        
        self.game.solution = self.bfs_algorithm(initial_state, goal_state, possible_moves)
        return self.game.solution
    

    def auto_play_dfs(self):
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
            valid_moves=possible_moves  
        )
        

        goal_state = Simulation(0, None, None, None, None)
        
        self.game.solution = self.dfs_algorithm(initial_state, goal_state, possible_moves)
        return self.game.solution
    
    def auto_play_iterative_deepning(self):
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
            valid_moves=possible_moves  
        )
        

        goal_state = Simulation(0, None, None, None, None)
        
        start_time = time.time()
        elapsed_time = 0
        num_moves = 0
        num_states = 0
        memory_used = 0

        for i in range(1, 9999999, 1):
                goal_state_reached = [False]
                a = self.iterative_deepning_algorithm(initial_state, goal_state, possible_moves, i, goal_state_reached)
                if(goal_state_reached[0] == True):
                    (moves, tmp, mem, estados) = a
                    elapsed_time += tmp
                    memory_used += mem
                    num_states += estados
                    num_moves += len(moves)
                    write_to_csv(
                        f"iterative_deepening_metrics_{self.game.level}.csv",
                        [[num_moves, num_states, memory_used, elapsed_time]],
                        headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                    )
                    

                
                    print("FINAL Goal state reached")
                    print(f"Number of states visited: {num_states}")
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")
                    print(f"Memory used: {memory_used} bytes")
                    self.game.solution = moves
                    return moves
                else:
                    (tempo, memory, states) = a
                    elapsed_time += tempo
                    memory_used += memory
                    num_states += states
    
    def astar_algorithm(self, current_state, goal_state, possible_moves):
        start_time = time.time()
        counter = 0
    
        if not hasattr(current_state, 'valid_moves') or current_state.valid_moves is None:
            current_state.valid_moves = possible_moves
    
        open_set = []
        closed_set = set()
        g_score = {current_state: 0}
        f_score = {current_state: self.heuristic_astar(current_state)}
        parents = {current_state: (None, None, 0)}  
    
        heapq.heappush(open_set, (f_score[current_state], current_state))
    
        closest_f_score = f_score[current_state]
    
        while open_set:
            _, current = heapq.heappop(open_set)
    
            if current.reds == 0:
                elapsed_time = time.time() - start_time
                moves = self.reconstruct_move(parents, current)
                num_moves = len(moves)
                memory_used = sys.getsizeof(open_set) + sys.getsizeof(closed_set) + sys.getsizeof(parents)
    
                if(self.game.player_type != 6):
                    write_to_csv(
                        f"astar_lv{self.game.level}.csv",
                        [[num_moves, counter, memory_used, elapsed_time]],
                        headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                    )
                else:
                    write_to_csv(
                        f"special_astar_lv{self.game.level}.csv",
                        [[num_moves, counter, memory_used, elapsed_time]],
                        headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
                    )

                print("Goal state reached")
                print(f"Number of states visited: {counter}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Memory used: {memory_used} bytes")
                return moves
    
            closed_set.add(current)
            counter += 1
    
            if f_score[current] < closest_f_score:
                closest_f_score = f_score[current]
    
            _, _, current_depth = parents[current]
    
            if not hasattr(current, 'valid_moves') or current.valid_moves is None:
                current.valid_moves = self.find_possible_moves(current.game)
    
            for move in current.valid_moves:
                (block, (row, col), simulation) = move
                neighbor = self.simulate_move(current.game, block, (row, col))
    
                tentative_g_score = g_score[current] + self.cost(current, move)
    
                if tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
    
                parents[neighbor] = (current, move, current_depth + 1)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic_astar(neighbor)
    
                if neighbor in closed_set:
                    continue
    
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                print(f"Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")
    
        elapsed_time = time.time() - start_time
        memory_used = sys.getsizeof(open_set) + sys.getsizeof(closed_set) + sys.getsizeof(parents)
    
        write_to_csv(
            f"astar_algorithm_metrics_{self.game.level}.csv",
            [[0, counter, memory_used, elapsed_time]],
            headers=["Number of Moves", "Number of States", "Memory (bytes)", "Time (seconds)"]
        )
    
        print("Goal state not found")
        print(f"Number of states visited: {counter}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Memory used: {memory_used} bytes")
        return None
    

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
        
        goal_state = Simulation(0, None, None, None, None)
        
        self.game.solution = self.astar_algorithm(current_state, goal_state, possible_moves)
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
        elif self.game.player_type == 6:
            return self.auto_play_astar()
        else:
            print("Invalid mode. Please select 'greedy', 'astar', or 'bfs'.")
            return None