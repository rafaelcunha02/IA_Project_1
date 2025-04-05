import pygame
from game import Game
from constants import BACKGROUND_COLOR, WINDOW_WIDTH, WINDOW_HEIGHT, screen

class Menu:
    def __init__(self):
        self.title_font = pygame.font.SysFont('Arial', 48)
        self.font = pygame.font.SysFont('Arial', 24)
        self.level_options = ["Level 1", "Level 2", "Level 3", "Infinite Mode"]
        self.player_options = ["Human", "Greedy AI", "Brute Force BFS AI", "Brute Force DFS AI", "Iterative Deepening AI", "AStar", "Back (B)"]
        self.current_option = 0
        self.current_menu = "level"  # "level" or "player"
        self.selected_level = None

    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        title_text = self.title_font.render("Block Blast", True, (255, 255, 255))
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, WINDOW_HEIGHT // 2 - 150))

        options = self.level_options if self.current_menu == "level" else self.player_options

        for i, option in enumerate(options):
            color = (255, 255, 0) if i == self.current_option else (255, 255, 255)
            option_text = self.font.render(option, True, color)
            screen.blit(option_text, (WINDOW_WIDTH // 2 - option_text.get_width() // 2, WINDOW_HEIGHT // 2 - 50 + i * 50))

        pygame.display.flip()

    def handle_key_event(self, key):
        if key == pygame.K_DOWN:
            self.current_option = (self.current_option + 1) % len(self.level_options if self.current_menu == "level" else self.player_options)
        elif key == pygame.K_UP:
            self.current_option = (self.current_option - 1) % len(self.level_options if self.current_menu == "level" else self.player_options)
        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            return self.select_option()
        elif key == pygame.K_b and self.current_menu == "player":
            self.current_menu = "level"
            self.current_option = self.selected_level
        return None

    def handle_mouse_event(self, pos):
        options = self.level_options if self.current_menu == "level" else self.player_options
        for i, option in enumerate(options):
            option_text = self.font.render(option, True, (255, 255, 255))
            option_rect = option_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50 + i * 50))
            if option_rect.collidepoint(pos):
                self.current_option = i
                return self.select_option()
        return None
    

    def handle_mouse_motion(self, pos):
        options = self.level_options if self.current_menu == "level" else self.player_options
        for i, option in enumerate(options):
            option_text = self.font.render(option, True, (255, 255, 255))
            option_rect = option_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50 + i * 50))
            if option_rect.collidepoint(pos):
                self.current_option = i
                

    def select_option(self):
        if self.current_menu == "level":
            self.selected_level = self.current_option
            self.current_option = 0
            self.current_menu = "player"
        else:
            if self.current_option == len(self.player_options) - 1:  # "Back" option
                self.current_menu = "level"
                self.current_option = self.selected_level
            else:
                player_type = self.current_option
                return self.start_game(self.selected_level, player_type)
        return None

    def start_game(self, level, player_type):
        # Initialize the game with the selected level and player type
        game = Game(level + 1, player_type)
        game.player_type = player_type  # Assuming Game class can handle player_type
        print(f"Starting game with level {level + 1} and player type {player_type}")
        return game