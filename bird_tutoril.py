import pygame
import neat
import time
import os 
import random
import math
import pickle


WIN_WIDTH = 600
WIN_HEIGHT = 600
score=0
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]

PIPE_IMG = pygame.transform.scale(pygame.image.load(os.path.join("imgs","pipe.png")),(50,10))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")),(WIN_WIDTH,WIN_HEIGHT))
ANIME_IMG =pygame.transform.scale(pygame.image.load(os.path.join("imgs","anime.png")),(50,50))
MAZE_IMG = pygame.transform.scale(pygame.image.load(os.path.join("imgs","maze.png")),(WIN_WIDTH,WIN_HEIGHT))

gen = 0



class Bird:
   
    

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.imgs = ANIME_IMG
        self.up_speed=3
        self.right_speed=10
        self.left_speed=-10
        self.image=ANIME_IMG
    
    
    def jump(self):
        pass
    
    def move_down(self):
        self.y=self.y+self.up_speed
        
        

    def move_up(self):
        self.y=self.y-self.up_speed
        
    
    def move_left(self):
        self.x=self.x+self.left_speed
        if (self.x<10):
            self.x=10
        
    
    def move_right(self):
        self.x=self.x+self.right_speed
        if (self.x>590):
            self.x=590
        

    
    
    def draw(self,win):
        win.blit(ANIME_IMG,(self.x,self.y))



class pipe:

    def __init__(self):
        self.imgs = PIPE_IMG
        self.y=500
        self.x=100
        self.up_speed=3
        self.image=PIPE_IMG

    def move_up(self):
        
        self.y=self.y-self.up_speed
        if self.y<=100:
            self.y=590
            self.x=random.random()*600
    
    def draw(self,win):
        win.blit(self.imgs,(self.x,self.y))

    def collision(self,birds):
        #distance = pow(pow((self.x - pipes.x),2)+pow((self.y - pipes.x),2),0.5)
        #print(self.x,self.y,pipes.x,pipes.y)
        if birds.x+50>self.x and birds.x-50<self.x and self.y-birds.y<50 and self.y-birds.y>0:
            birds.up_speed = -self.up_speed
            return True
        else:
            birds.up_speed = self.up_speed
            return False



def draw_window(win,birds,pipes,gen):
    if gen == 0:
        gen = 1
    win.blit(BG_IMG,(0,0))
    for bird in birds:
        bird.draw(win)
    for pipe in pipes:
        pipe.draw(win)
        
    pygame.display.update()


    


def eval_genomes(genomes, config):
    global WIN, gen
    win = WIN
    gen += 1

    birds = []
    ge=[]
    nets=[]

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(random.random()*600,10))
        ge.append(genome)
    
    pipes = [pipe()]
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()
    #timer_event = pygame.USEREVENT + 1
    #pygame.time.set_timer(timer_event, 250)
    run = True
    while run and len(birds) > 0:
        clock.tick(200)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run= False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        
        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move_down()
            
            

            output = nets[birds.index(bird)].activate((bird.y,bird.x, abs(bird.x - pipes[pipe_ind].x), abs(bird.y - pipes[pipe_ind].y)))

            if output[0] > 0.5:
                #ge[x].fitness += 0.1  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                bird.move_right()

            if output[1] > 0.5:
                #ge[x].fitness += 0.1  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                bird.move_left()

      
        
            if pipes[0].collision(bird):
                ge[x].fitness += 0
                

        
        pipes[0].move_up()
        for x, bird in enumerate(birds):
            if bird.y < 10 or bird.y > 590:
                ge[x].fitness -= 20
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))
            

        
                

        draw_window(WIN, birds, pipes, gen)
            
        
    

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

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)
    with open('company_data.pkl', 'wb') as output:
        pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)
    
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)




