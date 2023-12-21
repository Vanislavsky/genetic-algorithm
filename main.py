import numpy as np
import genetic_algorithm

NUM_EPOCH = 1000
POPULATION_SIZE = 200
SELECTION_SIZE = 60
P_MUTATION_IND = 0.3
P_MUTATION_GEN = 0.2
K_POINT = 100


peoples_coeff = []
fin = open("input.txt", "r")
n = int(fin.readline())
tasks = np.array(list((map(int, fin.readline().split()))))
times = np.array(list(map(float, fin.readline().split())))
m = int(fin.readline())
for i in range(m):
    peoples_coeff.append(list(map(float, fin.readline().split())))
peoples_coeff = np.array(peoples_coeff)

ga = genetic_algorithm.Generic_algorithm(
    tasks_levels=tasks,
    count_tasks=n,
    count_engineer=m,
    population_size=POPULATION_SIZE,
    time_array=times,
    coeff_array=peoples_coeff,
    selection_size=SELECTION_SIZE,
    p_mutation_ind=P_MUTATION_IND,
    p_mutation_gen=P_MUTATION_GEN,
    k_point=K_POINT)


for i in range(NUM_EPOCH):
    print("Epoch: ", i)
    ga.step()

print(ga.best_individuals(1))
