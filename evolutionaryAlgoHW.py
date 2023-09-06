import numpy as np

# Link to Google collab:
# https://colab.research.google.com/drive/1gzpmOrrPD4rJycyWoocMvSdIbPwe2fNG?usp=sharing

# Integer to gray code
def int_to_gray(n, bits):
    n = n ^ (n >> 1)
    return format(n, '0' + str(bits) + 'b')

# Gray code to integer
def gray_to_int(g):
    n = int(g, 2)
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

# Fitness
def fitness(x):
    return x ** 2 # objective function to minimize

# One-point crossover
def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1) # define the point where the crossover will ocurr
    child1 = parent1[:crossover_point] + parent2[crossover_point:] # slices the parent strings at the crossover point and combines them
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# One-point mutation
def one_point_mutation(child):
    mutation_point = np.random.randint(0, len(child))
    mutated = list(child)
    mutated[mutation_point] = '1' if child[mutation_point] == '0' else '0' # pick a point to cause a mutation, a flip of number
    return ''.join(mutated)

# Tournament selection
def binary_tournament(population):
    idx1, idx2 = np.random.choice(len(population), 2, replace=False) # get random index 
    if fitness(gray_to_int(population[idx1])) < fitness(gray_to_int(population[idx2])): # check if the fit of one of the choices is better than the other
        return population[idx1]
    else:
        return population[idx2]

population_size = 20
bits = 9  # Bot size in order to represent numbers in range [-255, 255]
population = [int_to_gray(np.random.randint(-255, 256), bits) for _ in range(population_size)]

# Parameters
generations = 100
mutation_rate = 0.1

# Genetic Algorithm
for gen in range(generations):
    new_population = []
    for _ in range(population_size // 2): # run half the times of the population as two children will be generated
        parent1 = binary_tournament(population) # selection of parents
        parent2 = binary_tournament(population)

        child1, child2 = one_point_crossover(parent1, parent2) # crossover

        if np.random.rand() < mutation_rate: # establish a rate for mutations
            child1 = one_point_mutation(child1)

        if np.random.rand() < mutation_rate:
            child2 = one_point_mutation(child2)

        new_population.extend([child1, child2])

    # Find the best individual
    best_individual = min(population, key=lambda x: fitness(gray_to_int(x)))
    print(f"Generation {gen + 1}: Best individual = {gray_to_int(best_individual)}, Fitness = {fitness(gray_to_int(best_individual))}")

    population = new_population

best_solution = min(population, key=lambda x: fitness(gray_to_int(x)))
print(f"Best solution after {generations} generations: x = {gray_to_int(best_solution)}, f(x) = {fitness(gray_to_int(best_solution))}")