import os
import neat
import visualize

class robot():
    #initialize all robot attributes
    def __init__(self, id, body):
        self.id = id
        self.distance = 0
        self.height = body.height
        self.momentum = 0
        self.rotation = 0
        for joint in body.joints:
            self.joint.angle = 0
            self.joint.angular_velocity = 0
        for part in body.parts:
            self.part.touch = True
        self.inputs = ['height', 'momentum', 'rotation', 'joint.angle', 'joint.angular_velocity', 'part.touch']

    #method to update distance
    def get_distance(self):
        return distance
    
    #method to get inputs for neural network
    def get_input(self):
        senses = []
        for input in self.inputs:
            senses.append(get_input)
        return senses
    
    #robot carry out the outputs of the NN
    def act(self, outputs):
        for output in outputs:
            set_joint_speed(output)
        
    
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #wait for robot to run and die, continously call robot.get_input(), forward pass, and robot.act()
        net.activate(robot)
        genome.fitness = robot.distance


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(30))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

#run neuroevolution with the pre-defined config file
run('config.md')