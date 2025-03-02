import pygame
import sys
from pygame.locals import *
from game import Game
from menu import Menu
from constants import BACKGROUND_COLOR, title_font, screen, WINDOW_WIDTH, WINDOW_HEIGHT, clock

pygame.init()


def main():
    menu = Menu()
    current_block = None
    in_menu = True
    
    last_mouse_pos = None
    running = True
    while running:
        if in_menu:
            menu.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    game = menu.handle_key_event(event.key)
                    if game:
                        in_menu = False
                elif event.type == pygame.MOUSEMOTION:
                    game = menu.handle_mouse_event(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    game = menu.handle_mouse_event(event.pos)
                    if game:
                        in_menu = False
        else:
            screen.fill(BACKGROUND_COLOR)
            is_in_go_to_menu = False
            mouse_pos = pygame.mouse.get_pos()  # Get the current mouse position

            for event in pygame.event.get():
                if event.type in (MOUSEMOTION, MOUSEBUTTONDOWN, MOUSEBUTTONUP):
                    if game.check_mouse_in_go_to_menu(event.pos):
                        is_in_go_to_menu = True

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
                                    if game.check_game_over():
                                        game.game_over = True
                            else:
                                current_block.reset_position()
                                
                            current_block.stop_drag()
                            current_block = None
                            # Clear preview when we're done dragging
                            game.current_grid_pos = None

                        if game.check_mouse_in_go_to_menu(event.pos):
                            in_menu = True
                            print("Go to menu")

                # Handle key events
                if event.type == KEYDOWN:
                    if event.key == K_r and game.game_over:
                        game.reset()
                    elif event.key == K_m:
                        in_menu = True
            
            # Check if the mouse is in the "Go to menu" area
            if game.check_mouse_in_go_to_menu(mouse_pos):
                is_in_go_to_menu = True

            # Draw the game
            game.draw_grid()
            
            # Draw placement preview if we're dragging a block
            if current_block and game.current_grid_pos:
                game.draw_placement_preview(current_block)
                
            game.draw_blocks()
            game.draw_score()

            if is_in_go_to_menu:
                game.draw_go_to_menu_highlighted()
            else:
                game.draw_go_to_menu()
            
            if game.game_over:
                game.draw_game_over()
            
            title_text = title_font.render("Block Blast", True, (255, 255, 255))
            screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))
            
            pygame.display.flip()
            clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()