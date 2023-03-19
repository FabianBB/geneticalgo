import numpy as np
import random
from time import sleep
import numpy.random as npr
import matplotlib.pyplot as plt

DEBUG = True

N_GEN = 500  # todo experiment
POP_SIZE = 100  # todo experiment
N_ITEMS = 10
MAX_ITEM_VALUE = 10
WEIGHT_LIMIT = 50  # todo experiment
KNAPSACK = True

# --- KNAPSACK functions start ---

# generate n (=N_ITEMS) items as the collection of items to choose from for each possible solution knapsack
# each item has a weight from 1...10 and a value from 1...MAX_ITEM_VALUE
def generateItems():
    items = []
    for i in range(N_ITEMS):
        weight = random.randint(1, 10)
        value = random.randint(1, MAX_ITEM_VALUE)
        items.append([weight, value])
    return items


# generate a population of individuals (i.e. chromosomes) where each individual represents a possible solution for
#  the knapsack problem
# each individual selects a random nr of items to include in the sack, indicating each item's inclusion or not by 1 or 0
def createPopKnapsack(items):
    pop = []
    for i in range(POP_SIZE):
        individual = [random.choice([0, 1]) for _ in items]
        pop.append(individual)
    return pop


# define fitness function for knapsack problem as the sum of all items' values in the sack if the total weight is
#  within limit, otherwise 0
def fitnessKnapsack(individual):
    total_weight = 0
    total_value = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            total_weight += items[i][0]
            if total_weight > WEIGHT_LIMIT:
                return 0
            total_value += items[i][1]
    return total_value


# do simple one-point crossover halfway
# so one offspring gets first half of one parent and second half from other parent,
# and the second offspring the other way around
def onePointCx(p1, p2):
    o1 = p1[:(N_ITEMS // 2)] + p2[(N_ITEMS // 2):]
    o2 = p2[:(N_ITEMS // 2)] + p1[(N_ITEMS // 2):]
    return o1, o2


# do simple mutation by deciding to include or exclude one random item by flipping their inclusion bit
def mutateKnapsack(ind):
    idx = np.random.randint(0, N_ITEMS-1)
    if ind[idx] == 0:
        ind[idx] = 1
    else:
        ind[idx] = 0
    return ind

# --- KNAPSACK functions end ---

# generate n random points in a unit circle
def generatePoints(n):
    points = np.random.rand(n, 2)
    points = points * 2 - 1
    return points

# create distance matrix
def createMat(points):
    n = len(points)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i][j] = np.linalg.norm(points[i] - points[j])
    return mat


if not KNAPSACK:
    problem = createMat(generatePoints(10))

# Genetic Algo

# todo small error in comment: n = 10 in generate points, but n here is how many individuals (permutations) in population
# create a population of n chromosomes, the chromosome being a list of n indices representing the cities generated
# by generatePoints
def createPop(n):
    pop = []

    # generate permutations of the indices
    for i in range(n):
        pop.append(np.random.permutation(len(problem)))

    return pop


# fitness function for TSP is just thee total distance of the path
def fitnessTSP(c):
    val = 0
    # make sure to not go out of bounds
    for i in range(len(c) - 1):
        val += problem[c[i]][c[i + 1]]
    # in my thesis TSP is a cycle so here I add the distance from last index to first
    val += problem[c[-1]][c[0]]
    if val < 1:
        print("Below 1 fitness score for: ")
        print(c)
    return val


# # select N/2 parents using proportional selection
# def selectParents(pop):
#     print(pop)
#     # calculate population fitness
#     total = 0
#     for c in pop:
#         total += fitness(c)
#
#     # calc prob of each chromomsome
#     probs = []
#     for c in pop:
#         probs.append(fitness(c) / total)
#         print(probs) if DEBUG else None
#
#     print(sum(probs)) if DEBUG else None
#     print(probs) if DEBUG else None
#     print(pop) if DEBUG else None
#
#     # select N/2 parents
#     parents = random.sample(pop, k=(max(50, int(len(pop) / 2))))
#     return parents

# function above didnt work quite well so i slightly modified this one
# from https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
# TODO this selection works based on maximizing a fitness function
#  because the bigger the fitness function the bigger the probaility of picking
def selectParents(population, fitFunc):
    totalf = sum([fitFunc(c) for c in population])
    selection_probs = [fitFunc(c) / totalf for c in population]

    idx = npr.choice(len(population), p=selection_probs, replace=False, size=max(50, int(len(population) / 2)))
    return [population[i] for i in idx]


# apply mutation with small chance (chance applied in the general algorith flow)
# swapping 2 indices is nice since it does not change the solution drastically but may perturb the solution enough to
# create better offspring in the future
# this comes from research I did on TRP
def mutate(c):
    # swap two random indices
    i = np.random.randint(0, len(c))
    j = np.random.randint(0, len(c))
    c[i], c[j] = c[j], c[i]
    return c


def cx2(p1, p2):
    # TODO error in TSP crossover function?
    o1 = []
    o2 = []

    # select first bit from other parent
    o1.append(p2[0])
    o2.append(p1[0])

    while len(o1) < len(p1):
        # find index of last bit in o1 in p1
        index = np.where(p1 == o1[-1])

        # add bit at index in p2 to o1
        o1.append(p2[index][0])
    print(o1) if DEBUG else None

    while len(o2) < len(p2):
        index = np.where(p2 == o2[-1])

        o2.append(p1[index][0])
    print(o2) if DEBUG else None

    return o1, o2


# genetic algo
# todo should we maximize fitness for both problems? easier to calculate probabilities?
# TODO add elitism
def GA(pop, cxFunc, mutFunc, fitFunc, maxFit, cxProb, mutProb):

    # TODO generate population here
    avgFitnessPerGen = []

    for i in range(N_GEN):
        # select parents

        parents = selectParents(pop, fitFunc)

        # create offspring
        offspring = []
        for i in range(0, len(parents) - 2, 2):

            # crossover based on crossover rate
            if np.random.rand() < cxProb:
                o1, o2 = cxFunc(parents[i], parents[i + 1])
            else:
                o1, o2 = parents[i], parents[i + 1]

            # mutate based on mutation rate
            if np.random.rand() < mutProb:
                o1 = mutFunc(o1)
                o2 = mutFunc(o2)

            offspring.append(o1)
            offspring.append(o2)

        # replace old population with offspring
        pop = offspring
        # duplicate every chromosome in population
        pop = pop + pop

        print(pop) if DEBUG else None
        # print average fitness of population
        avgFit = sum([fitFunc(c) for c in pop]) / len(pop)
        avgFitnessPerGen.append(avgFit)
        if DEBUG:
            print("Average fitness: ", avgFit)

        # todo best fitness for knapsack is max but best for tsp is min, need arg for that

        # keep track of minimum fitness
        # if min([fitFunc(c) for c in pop]) < minF:
        #     minF = min([fitFunc(c) for c in pop])
        #     print("New min: ", minF)

        # keep track of maximum fitness
        maxOfPop = max([fitFunc(c) for c in pop])
        if maxOfPop > maxFit:
            maxFit = maxOfPop
            print("New max: ", maxFit)

    # final min fitness
    # print("Min fitness: ", minF)
    # # find best chromosome
    # best = pop[0]
    # for c in pop:
    #     if fitFunc(c) < fitFunc(best):
    #         best = c

    plt.plot(avgFitnessPerGen)
    plt.ylabel('Avg. fitness score')
    plt.xlabel('Generation')
    plt.show()

    print("Max fitness: ", maxFit)
    # find best chromosome
    best = pop[0]
    for c in pop:
        if fitFunc(c) > fitFunc(best):
            best = c

    return best


# run GA

if KNAPSACK:
    items = generateItems()
    if DEBUG:
        print("items:")
        print(items)
    pop = createPopKnapsack(items)
    best = GA(pop, cxFunc=onePointCx, mutFunc=mutateKnapsack, fitFunc=fitnessKnapsack,
              maxFit=0, cxProb=0.6, mutProb=0.05)
    # todo in nature: cx rate: 0.4 to 0.6, mut rate: 0.01 to 0.02
    print("best: ", best)
    print(fitnessKnapsack(best))
else:
    pop = createPop(100)
    # TODO attention: need to change fitness function approach to maximizing for TSP
    best = GA(pop, cx2, mutate, fitnessTSP, maxFit=100000, cxProb=1, mutProb=0.05)
    print("best: ", best)
    print(fitnessTSP(best))
