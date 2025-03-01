import pygame
import sys
from pygame.locals import *
from game import Game
from menu import Menu
from constants import BACKGROUND_COLOR, title_font, screen, WINDOW_WIDTH, WINDOW_HEIGHT, clock

pygame.init()

def main():
    game = Game()
    menu = Menu()
    current_block = None
    in_menu = True
    
    running = True
    while running:
        if in_menu:
            menu.draw()
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_s:
                        in_menu = False
        else:
            screen.fill(BACKGROUND_COLOR)
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    
                if not game.game_over:
                    if event.type == MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left mouse button
                            # Check if we clicked on a block
                            for block in game.blocks:
                                if block.is_point_inside(event.pos):
                                    current_block = block
                                    current_block.start_drag(event.pos)
                                    break
                                    
                    elif event.type == MOUSEMOTION:
                        if current_block:
                            current_block.update_position(event.pos)
                            # Update placement preview while dragging
                            game.update_placement_preview(current_block)
                            
                    elif event.type == MOUSEBUTTONUP:
                        if event.button == 1 and current_block:
                            # Try to place the block on the grid
                            if game.try_place_block(current_block):
                                game.blocks.remove(current_block)
                                
                                # If all blocks are placed, generate new ones
                                if not game.blocks:
                                    game.blocks = game.generate_blocks()
                                    
                                    # Check if game is over
                                    if game.check_game_over():
                                        game.game_over = True
                            else:
                                current_block.reset_position()
                                
                            current_block.stop_drag()
                            current_block = None
                            # Clear preview when we're done dragging
                            game.current_grid_pos = None
                
                # Restart game when R is pressed
                if event.type == KEYDOWN:
                    if event.key == K_r and game.game_over:
                        game.reset()
                    elif event.key == K_m and game.game_over:
                        in_menu = True
            
            # Draw the game
            game.draw_grid()
            
            # Draw placement preview if we're dragging a block
            if current_block and game.current_grid_pos:
                game.draw_placement_preview(current_block)
                
            game.draw_blocks()
            game.draw_score()
            
            if game.game_over:
                game.draw_game_over()
            
            # Draw title
            title_text = title_font.render("Block Blast", True, (255, 255, 255))
            screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))
            
            pygame.display.flip()
            clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()