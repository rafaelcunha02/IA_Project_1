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
        level1_text = self.font.render("Press 1 for Level 1", True, (255, 255, 255))
        level2_text = self.font.render("Press 2 for Level 2", True, (255, 255, 255))
        level3_text = self.font.render("Press 3 for Level 3", True, (255, 255, 255))
        infinite_text = self.font.render("Press I for Infinite Mode", True, (255, 255, 255))
        
        screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 
                                 WINDOW_HEIGHT // 2 - 150))
        screen.blit(start_text, (WINDOW_WIDTH // 2 - start_text.get_width() // 2, 
                                 WINDOW_HEIGHT // 2 - 50))
        screen.blit(level1_text, (WINDOW_WIDTH // 2 - level1_text.get_width() // 2, 
                                  WINDOW_HEIGHT // 2))
        screen.blit(level2_text, (WINDOW_WIDTH // 2 - level2_text.get_width() // 2, 
                                  WINDOW_HEIGHT // 2 + 50))
        screen.blit(level3_text, (WINDOW_WIDTH // 2 - level3_text.get_width() // 2, 
                                  WINDOW_HEIGHT // 2 + 100))
        screen.blit(infinite_text, (WINDOW_WIDTH // 2 - infinite_text.get_width() // 2, 
                                    WINDOW_HEIGHT // 2 + 150))
        pygame.display.flip()