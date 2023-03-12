import numpy as np
import random
from time import sleep
import numpy.random as npr

DEBUG = False


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


problem = createMat(generatePoints(10))


# Genetic Algo

# create a population of n chromosomes, the chromosome being a list of n indices representing the cities generated
# by generatePoints
def createPop(n):
    pop = []

    # generate permutations of the indices
    for i in range(n):
        pop.append(np.random.permutation(len(problem)))

    return pop


# fitness function for TSP is just thee total distance of the path
def fitness(c):
    val = 0
    # make sure to not go out of bounds
    for i in range(len(c) - 1):
        val += problem[c[i]][c[i + 1]]
    # in my thesis TSP is a cycle so here I add the distance from last index to first
    val += problem[c[-1]][c[0]]
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
def selectParents(population):
    totalf = sum([fitness(c) for c in population])
    selection_probs = [fitness(c) / totalf for c in population]

    idx = npr.choice(len(population), p=selection_probs, replace=False, size=max(50, int(len(population) / 2)))
    return [population[i] for i in idx]


# apply mutation with small chance
# swapping 2 indices is nice since it does not change the solution drastically but may perturb the solution enough to
# create better offspring in the future
# this comes from research I did on TRP
def mutate(c):
    if np.random.rand() < 0.05:
        # swap two random indices
        i = np.random.randint(0, len(c))
        j = np.random.randint(0, len(c))
        c[i], c[j] = c[j], c[i]
    return c


def cx2(p1, p2):
    o1 = []
    o2 = []

    # select first bit from other parent
    o1.append(p2[0])
    o2.append(p1[0])

    while len(o1) < len(p1):
        # find index of last bit in o1 in p1
        index = np.where(p1 == o1[-1])

        # add bit at index in p2 to o1
        o1.append(p2[index])
        print(o1) if DEBUG else None

    while len(o2) < len(p2):
        index = np.where(p2 == o2[-1])

        o2.append(p1[index])
        print(o2) if DEBUG else None

    return p1, p2


# genetic algo
def GA(pop, ngen):
    minF = 100000
    for i in range(ngen):
        # select parents
        parents = selectParents(pop)

        # create offspring
        offspring = []
        for i in range(0, len(parents) - 2, 2):
            # crossover
            o1, o2 = cx2(parents[i], parents[i + 1])

            # mutate
            o1 = mutate(o1)
            o2 = mutate(o2)

            offspring.append(o1)
            offspring.append(o2)

        # replace old population with offspring
        pop = offspring
        # duplicate every chromosome in population
        pop = pop + pop

        print(pop) if DEBUG else None
        # print average fitness of population
        print("Average fitness: ", sum([fitness(c) for c in pop]) / len(pop))
        # keep track of minimum fitness
        if min([fitness(c) for c in pop]) < minF:
            minF = min([fitness(c) for c in pop])
            print("New min: ", minF)

    #print minF
    print("Min fitness: ", minF)
    # find best chromosome
    best = pop[0]
    for c in pop:
        if fitness(c) < fitness(best):
            best = c

    return best


# run GA
pop = createPop(100)


best = GA(pop, 500)
print("best: ", best)
print(fitness(best))
