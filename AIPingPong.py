import pygame
import os
from os import path
import neat
import random
import math

pygame.init()

clock = pygame.time.Clock()
fps = 60

#create display window
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption('Kamala 2024')

#game variables
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
    
class Ball():
    def __init__(self, color, x, y, radius):
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius
        self.x_vel = (random.random() - 0.5) * 4
        self.y_vel = math.sqrt(18 - self.x_vel * self.x_vel)
        
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def update(self):
        
        # Move the ball
        self.x += self.x_vel
        self.y += self.y_vel

        # Bounce off the left and right walls
        if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
            self.x_vel *= -1
        
        # If the ball touches the top or bottom, reset it
        if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
            self.reset()

    def hit(self):
        # Reverse vertical velocity when hit by paddle
        self.y_vel *= -1
        self.x_vel += random.uniform(-0.2, 0.2)
        
    def getRect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def reset(self):
        # Reset ball to the center and randomize its velocity
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2 
        self.x_vel = (random.random() - 0.5) * 4
        self.y_vel = math.sqrt(18 - self.x_vel * self.x_vel)

class Paddle():
    def __init__(self, color, x, y, width, height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = 5

    def moveRight(self):
        self.x += self.velocity
        if (self.x >= SCREEN_WIDTH - self.width):
            self.x = SCREEN_WIDTH - self.width

    def moveLeft(self):
        self.x -= self.velocity
        if (self.x <= 0):
            self.x = 0

    def draw(self):
        pygame.draw.rect(screen, self.color, pygame.Rect((self.x, self.y), (self.width, self.height)))

    def getRect(self):
        return pygame.draw.rect(screen, self.color, pygame.Rect((self.x, self.y), (self.width, self.height)))
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
class Opponent(Paddle):
    def __init__(self, color, x, y, width, height):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def update(self, ball):
        self.x = ball.x - self.width // 2
        if (self.x <= 0):
            self.x = 0
        if (self.x >= SCREEN_WIDTH - self.width):
            self.x = SCREEN_WIDTH - self.width

class Zone():
    def __init__(self):
        # Create the rectangle at the bottom of the screen
        self.rect = pygame.Rect(0, SCREEN_HEIGHT - 5, SCREEN_WIDTH, 5)

    def draw(self):
        # Draw the rectangle (zone) in red color
        pygame.draw.rect(screen, (255, 0, 0), self.rect)

    def getRect(self):
        return pygame.draw.rect(screen, (255, 0, 0), self.rect)


opponent = Opponent(RED, SCREEN_WIDTH // 2 - 15, 25, 40, 10)
ball = Ball(WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 5)
zone = Zone()
        
#game loop
def main(genomes, config):
    players = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append(Paddle(BLUE, SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 25, 40, 10))
        ge.append(genome)

    run = True
    while run:
        clock.tick(fps)
        screen.fill((0, 0, 0))

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        ball.update()
        ball.draw()
        zone.draw()

        # Collision with the opponent
        if ball.getRect().colliderect(opponent.getRect()):
            ball.hit()
            if ((ball.x_vel * ball.x_vel + ball.y_vel * ball.y_vel) <= 36):
                if (ball.y_vel < 0):
                    ball.y_vel -= 0.25
                else:
                    ball.y_vel += 0.25
                ball.x_vel += 0.25

        for index, player in enumerate(players):

            distance_to_ball = abs(player.x - ball.x)

            output = nets[index].activate([distance_to_ball, ball.y, (ball.x_vel * ball.x_vel + ball.y_vel * ball.y_vel)])
            if output[0] < 0.5:
                player.moveRight()
            elif output[0] > 0.5:
                player.moveLeft()


            # Collision with the player's paddle
            if ball.getRect().inflate(2,2).colliderect(player.getRect()):
                ball.hit()
                ge[index].fitness += 0.1

            ge[index].fitness += 0.01

            player.draw()
            # Check if the ball is out of bounds (on the bottom side)
            if ball.getRect().colliderect(zone.getRect()):
                if distance_to_ball < 20:  # Reward for being close to the ball
                    ge[index].fitness += 0.1
                elif distance_to_ball > 0.5 * SCREEN_WIDTH:
                    ge[index].fitness -= 10
                ge[index].fitness -= 1
                players.pop(index)
                nets.pop(index)
                ge.pop(index)
    

        # If no players are left, end the game
        if len(players) <= 0:
            run = False

        opponent.update(ball)
        opponent.draw()

        pygame.display.update()


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 30 generations.
    winner = p.run(main, 30)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)