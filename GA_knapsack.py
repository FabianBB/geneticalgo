import numpy as np
import random
from time import sleep
import numpy.random as npr
import matplotlib.pyplot as plt

DEBUG = False

N_GEN = 2000
POP_SIZE = 100  # todo experiment 100, 200, 300
N_ITEMS = 10
MAX_ITEM_VALUE = 10
WEIGHT_LIMIT = 50  # todo experiment 40 50 60 or 50 60 70
KNAPSACK = False

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

def twoPointCx(p1, p2):
    # only for length 10 for now
    o1 = p1[:3] + p2[3:6] + p1[6:]
    o2 = p2[:3] + p1[3:6] + p2[6:]
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

# Genetic Algo

# create a population of size POP_SIZE chromosomes, the chromosome being a list of n indices representing the cities generated
# by generatePoints
def createPop(problem):
    pop = []

    # generate permutations of the indices
    for i in range(POP_SIZE):
        pop.append(np.random.permutation(len(problem)))

    return pop


# fitness function for TSP is just the inverse of total distance of the path
# to be able to maximize it
def fitnessTSP(c):
    val = 0
    # make sure to not go out of bounds
    for i in range(len(c) - 1):
        val += problem[c[i]][c[i + 1]]
    # in my thesis TSP is a cycle so here I add the distance from last index to first
    val += problem[c[-1]][c[0]]
    return 1 / val


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
def selectParents(population, fitFunc):
    totalf = np.sum([fitFunc(c) for c in population])
    selection_probs = [(fitFunc(c) / totalf) for c in population]
    # print(selection_probs)
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


# midway split crossover
def halfhalf(p1, p2):

    o1 = p1[0:len(p1) // 2]
    o2 = p2[0:len(p2) // 2]

    for val in p2:

        if not val in o1:
            o1 = np.concatenate((o1, [val]))

    for val in p1:
        if not val in o2:
            o2 = np.concatenate((o2, [val]))

    return o1, o2

# order crossover by Davis
# Davis L. Applying adaptive algorithms to epistatic domains. IJCAI. 1985;85:162–164. [Google Scholar]
def ox(p1, p2):
    # initiliaze offspring as lists filled with -1
    o1 = [-1] * len(p1)
    o2 = [-1] * len(p2)

    # select random start and end indices
    start = np.random.randint(0, len(p1))
    end = np.random.randint(0, len(p1))

    # make sure start is smaller than end
    if start > end:
        start, end = end, start

    # copy the selected part of the parents to the offspring
    o1[start:end] = p1[start:end]
    o2[start:end] = p2[start:end]

    # fill the rest of the offspring with the remaining values from the parents
    # check for duplicates
    for i in range(len(p1)):
        if not p1[i] in o1:
            for j in range(len(o1)):
                if o1[j] == -1:
                    o1[j] = p1[i]
                    break

    for i in range(len(p2)):
        if not p2[i] in o2:
            for j in range(len(o2)):
                if o2[j] == -1:
                    o2[j] = p2[i]
                    break

    return o1, o2


# genetic algo
# fixed set is fixed problem initialization (e.g. item collection, cities)
# popGeneration initalizes the population based on that
def GA(fixedSet, popGeneration, cxFunc, mutFunc, fitFunc, maxFit, cxProb, mutProb, kElitist):

    pop = popGeneration(fixedSet)

    maxFitnessPerGen = []

    for i in range(N_GEN):
        # elitism with sorting like this takes too long, would need a better implementation
        # elitism: retain k best individuals (the elite) to carry over into next gen
        # if kElitist > 0:
        #     sorted_pop = sorted(pop, key=lambda ind: fitFunc(ind), reverse=True)
        #     elite = sorted_pop[:kElitist]
        # else:
        #     sorted_pop = pop
        #     elite = []

        # select parents for mating without elite
        # parents = selectParents(sorted_pop[kElitist:], fitFunc)

        # select parents for mating
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

        # replace old population with offspring + saved elite
        # pop = offspring + elite

        # replace old population with offspring
        pop = offspring
        # duplicate every chromosome in population
        pop = pop + pop

        print(pop) if DEBUG else None
        # print average fitness of population
        avgFit = sum([fitFunc(c) for c in pop]) / len(pop)
        if DEBUG:
            print("Average fitness: ", avgFit)

        # keep track of maximum fitness
        maxOfPop = max([fitFunc(c) for c in pop])
        maxFitnessPerGen.append(maxOfPop)
        if maxOfPop > maxFit:
            maxFit = maxOfPop
            print("New max: ", maxFit)

    # plt.plot(maxFitnessPerGen, color='b', label='max. fitness')
    # plt.ylabel('Max fitness of population')
    # plt.xlabel('Generation')
    # plt.show()

    print("Max fitness: ", maxFit)
    # find best chromosome
    best = pop[0]
    for c in pop:
        if fitFunc(c) > fitFunc(best):
            best = c

    return best, maxFitnessPerGen

def plotAvgOfExperiments(listOfExpResults, name):
    n_exp = len(listOfExpResults)
    avg_results = []
    # for each generation
    for i in range(N_GEN):
        # get the average of all experiments' max fitness
        sum = 0
        for j in range(n_exp):
            sum += listOfExpResults[j][i]
        avg_results.append(sum / n_exp)
    # plot
    plt.title("Avg. results of 10 exp-s")
    plt.plot(avg_results)
    plt.ylabel('Max fitness')
    plt.xlabel('Generation')
    plt.show()
    plt.savefig(fname=name)
    return avg_results


def plotFour(list1, list2, list3, list4):
    # plot three in same
    plt.title("mutation probability comparison")
    plt.plot(list1, color='r', label="max_w=40")
    plt.plot(list2, color='g', label="max_w=50")
    plt.plot(list3, color='b', label="max_w=60")
    # plt.plot(list4, color='y', label="max_w=60")
    plt.ylabel('Max fitness')
    plt.xlabel('Generation')
    plt.show()
    plt.savefig(fname="1p_vs_ox_comp.png")
    return

# run GA

items = generateItems()
print(items)

if KNAPSACK:
    # use one problem set for parameter experiment
    exps = []
    # run 10 experiments for configuration
    POP_SIZE = 100
    cxProb = 0.5
    mutProb = 0.025
    print("Experiment one-point crossover:")
    print("pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb))
    for i in range(10):
        best, generations = GA(items, createPopKnapsack, cxFunc=onePointCx, mutFunc=mutateKnapsack, fitFunc=fitnessKnapsack,
                    maxFit=0, cxProb=cxProb, mutProb=mutProb, kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "ONEPOINT|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".jpeg"
    cx1 = plotAvgOfExperiments(exps, name=plot_name)

    # POP_SIZE = 200
    # cxProb = 0.5
    mutProb = 0.05
    print("Experiment one-point crossover:")
    print("pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb))
    for i in range(10):
        best, generations = GA(items, createPopKnapsack, cxFunc=onePointCx, mutFunc=mutateKnapsack,
                               fitFunc=fitnessKnapsack,
                               maxFit=0, cxProb=cxProb, mutProb=mutProb, kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "ONEPOINT|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".jpeg"
    cx2 = plotAvgOfExperiments(exps, name=plot_name)

    # POP_SIZE = 300
    # cxProb = 0.5
    mutProb = 0.075
    print("Experiment one-point crossover:")
    print("pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb))
    for i in range(10):
        best, generations = GA(items, createPopKnapsack, cxFunc=onePointCx, mutFunc=mutateKnapsack,
                               fitFunc=fitnessKnapsack,
                               maxFit=0, cxProb=cxProb, mutProb=mutProb, kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "ONEPOINT|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".png"
    cx3 = plotAvgOfExperiments(exps, name=plot_name)

    plotFour(cx1, cx2, cx3, [])

else:
    problem = createMat(generatePoints(10))
    exps = []

    POP_SIZE = 100
    cxProb = 0.5
    mutProb = 0.025
    for i in range(10):
        best, generations = GA(problem, createPop, ox, mutate, fitnessTSP, maxFit=0, cxProb=cxProb, mutProb=mutProb, kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "TSP|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".png"
    cx1 = plotAvgOfExperiments(exps, name=plot_name)
    print("best: ", best)
    print(fitnessTSP(best))

    POP_SIZE = 200
    # cxProb = 0.7
    # mutProb = 0.025
    for i in range(10):
        best, generations = GA(problem, createPop, ox, mutate, fitnessTSP, maxFit=0, cxProb=cxProb, mutProb=mutProb,
                               kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "TSP|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".png"
    cx2 = plotAvgOfExperiments(exps, name=plot_name)
    print("best: ", best)
    print(fitnessTSP(best))

    POP_SIZE = 300
    # cxProb = 0.9
    # mutProb = 0.025
    for i in range(10):
        best, generations = GA(problem, createPop, ox, mutate, fitnessTSP, maxFit=0, cxProb=cxProb, mutProb=mutProb,
                               kElitist=0)
        exps.append(generations)
    print("best: ", best)
    print(fitnessKnapsack(best))
    plot_name = "TSP|pop_size=" + str(POP_SIZE) + "|cx_prob=" + str(cxProb) + "|mut_prob=" + str(mutProb) + ".png"
    cx3 = plotAvgOfExperiments(exps, name=plot_name)
    print("best: ", best)
    print(fitnessTSP(best))

    plotFour(cx1, cx2, cx3, [])
