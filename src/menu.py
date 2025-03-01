import pygame
from constants import BACKGROUND_COLOR, WINDOW_WIDTH, WINDOW_HEIGHT, screen

class Menu:
    def __init__(self):
        self.title_font = pygame.font.SysFont('Arial', 48)
        self.font = pygame.font.SysFont('Arial', 24)
    
    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        title_text = self.title_font.render("Block Blast", True, (255, 255, 255))
        start_text = self.font.render("Press S to Start", True, (255, 255, 255))
        
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 
                                 WINDOW_HEIGHT // 2 - 100))
        screen.blit(start_text, (WINDOW_WIDTH // 2 - start_text.get_width() // 2, 
                                 WINDOW_HEIGHT // 2))
        pygame.display.flip()