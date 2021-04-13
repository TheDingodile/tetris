import pygame
from helpers import Zero_dividor

class UI():
    def initilize(self):
        # Initialize the game engine
        pygame.init()

        size = (800, 800)
        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()

    def draw_step(self, game):    
        # Define some colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128)

        self.screen.fill(WHITE)

        self.action(game)
        if game.high_performance == 1:
            return

        for i in range(game.height):
            for j in range(game.width):
                pygame.draw.rect(self.screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
                if game.field[0,0,i,j] != 0:
                    pygame.draw.rect(self.screen, (120,120,120),
                                    [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

        if game.figure[0] is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in game.figure[0].image():
                        pygame.draw.rect(self.screen, game.figure[0].color,
                                        [game.x + game.zoom * (j + game.figure[0].x) + 1,
                                        game.y + game.zoom * (i + game.figure[0].y) + 1,
                                        game.zoom - 2, game.zoom - 2])
        for k in range(len(game.next_pieces[0])):
            the_figure = game.next_pieces[0][k]
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in the_figure.figure[0]:
                        pygame.draw.rect(self.screen, the_figure.color,
                                        [250 + game.x + game.zoom * j + 1,
                                        70*k + game.y + game.zoom * i + 1,
                                        game.zoom - 2, game.zoom - 2])    


        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(game.score), True, BLACK)
        text2 = font.render("Level: " + str(game.level), True, BLACK)
        text3 = font.render("Tetris rate: " + Zero_dividor(400*game.all_tetrises,game.all_lines), True, BLACK)
        text4 = font.render("Lines: " + str(game.all_lines), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        self.screen.blit(text2, [600, 0])
        self.screen.blit(text3, [600, 600])
        self.screen.blit(text4, [0, 600])

        if True is False:
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_game_over1, [25, 265])

        pygame.display.flip()

    pygame.quit()
    
    def action(self, game):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f and game.fps < 15*2**6:
                game.fps *= 2
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s and game.fps > 1/(15*2**6):
                game.fps /= 2
            if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                game.high_performance = 1 - game.high_performance
            if event.type == pygame.KEYDOWN and game.player1 == "human":
                if event.key == pygame.K_z:
                    game.rotate_clock()
                if event.key == pygame.K_x:
                    game.rotate_counter_clock()
                if event.key == pygame.K_DOWN:
                    game.pressing_down = True
                if event.key == pygame.K_LEFT:
                    game.go_side(-1)
                if event.key == pygame.K_RIGHT:
                    game.go_side(1)
                if event.key == pygame.K_SPACE:
                    game.go_space()
                if event.key == pygame.K_ESCAPE:
                    game.restart()

            if event.type == pygame.KEYUP and game.player1 == "human":
                    if event.key == pygame.K_DOWN:
                        game.pressing_down = False

