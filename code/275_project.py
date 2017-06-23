import gym
import numpy as np
import time, math, random, bisect, copy

class NeuralNetwork : 
    def __init__(self, structure):    
        self.structure = structure 
        self.weights = []
        self.biases = []
        self.fitness = 0.0
        for i in range(len(structure) - 1):
            self.weights.append( np.random.uniform(low=-1, high=1, size=(structure[i], structure[i+1])).tolist() )
            self.biases.append( np.random.uniform(low=-1, high=1, size=(structure[i+1])).tolist())

    def getOutput(self, input):
        output = input
        for i in range(len(self.structure)-1):
            output = np.reshape( np.matmul(output, self.weights[i]) + self.biases[i], (self.structure[i+1]))
            ########## Actuation Function ==> sigmoid linear tanh
            output = [sigmoid(x) for x in output]
            #output =  np.reshape(output, self.structure[i+1])
            #output = [np.tanh(x) for x in output]
            output =  np.reshape(output, self.structure[i+1])
            ##########
        return output

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x)) 

class Population :
    def __init__(self, populationCount, mutationRate, structure):
        self.structure = structure
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ NeuralNetwork(structure) for i in range(populationCount)]


    def crossoverAndMutate(self, nn1, nn2):
        
        child = NeuralNetwork(self.structure)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:
                        if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else :
                            child.weights[i][j][k] = nn2.weights[i][j][k]
                            
        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]

        return child


    def createNewGeneration(self, bestNN):    
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]));

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)
        

        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.crossoverAndMutate(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ");
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        #self.population.clear()
        self.population[:] = []
        self.population = nextGen




GAME = 'BipedalWalker-v2'
env = gym.make(GAME)
MAX_STEPS = 1000
MAX_GENERATIONS = 10000
POPULATION_COUNT = 100
MUTATION_RATE = 0.01

observation = env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
obsMin = env.observation_space.low
obsMax = env.observation_space.high
actionMin = env.action_space.low
actionMax = env.action_space.high
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 13, 8, 13, out_dimen])
bestNeuralNets = []

print("\nObservation\n--------------------------------")
print("Shape :", in_dimen, " \n High :", obsMax, " \n Low :", obsMin)
print("\nAction\n--------------------------------")
print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin,"\n")

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    minFit =  1000000
    maxFit = -1000000
    maxNeuralNet = None
    for nn in pop.population:
        observation = env.reset()
        totalReward = 0
        for step in range(MAX_STEPS):
            env.render()
            action = nn.getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break

        nn.fitness = totalReward
        minFit = min(minFit, nn.fitness)
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            maxNeuralNet = copy.deepcopy(nn);
        bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
    pop.createNewGeneration(maxNeuralNet)
print("Evolution Done.")





