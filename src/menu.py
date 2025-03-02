import pygame
from game import Game
from constants import BACKGROUND_COLOR, WINDOW_WIDTH, WINDOW_HEIGHT, screen

class Menu:
    def __init__(self):
        self.title_font = pygame.font.SysFont('Arial', 48)
        self.font = pygame.font.SysFont('Arial', 24)
        self.options = ["Level 1", "Level 2", "Level 3", "Infinite Mode"]
        self.current_option = 0

    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        title_text = self.title_font.render("Block Blast", True, (255, 255, 255))
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, WINDOW_HEIGHT // 2 - 150))

        for i, option in enumerate(self.options):
            color = (255, 255, 0) if i == self.current_option else (255, 255, 255)
            option_text = self.font.render(option, True, color)
            screen.blit(option_text, (WINDOW_WIDTH // 2 - option_text.get_width() // 2, WINDOW_HEIGHT // 2 - 50 + i * 50))

        pygame.display.flip()

    def handle_key_event(self, key):
        if key == pygame.K_DOWN:
            self.current_option = (self.current_option + 1) % len(self.options)
        elif key == pygame.K_UP:
            self.current_option = (self.current_option - 1) % len(self.options)
        elif key == pygame.K_RETURN:
            return self.select_option()
        elif key == pygame.K_KP_ENTER:
            return self.select_option()
        return None

    def handle_mouse_event(self, pos):
        for i, option in enumerate(self.options):
            option_text = self.font.render(option, True, (255, 255, 255))
            option_rect = option_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50 + i * 50))
            if option_rect.collidepoint(pos):
                self.current_option = i
                return self.select_option()
        return None

    def select_option(self):
        if self.current_option == 0:
            return Game(1)
        elif self.current_option == 1:
            return Game(2)
        elif self.current_option == 2:
            return Game(3)
        elif self.current_option == 3:
            return Game(0)
        return None