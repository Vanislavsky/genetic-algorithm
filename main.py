import numpy as np

class Generic_algorithm:
    def __init__(self,
                 tasks,
                 count_tasks,
                 count_engineer,
                 population_size,
                 time_array,
                 coeff_array,
                 size_selection,
                 p_mutation_ind: float,
                 p_mutation_gen: float,
                 k_point=1):
        self.tasks = tasks
        self.count_tasks = count_tasks
        self.count_engineer = count_engineer
        self.rng = np.random.default_rng()
        self.population_size = population_size
        self.time_array = time_array
        self.coeff_array = coeff_array
        self.size_selection = size_selection
        self.p_mutation_ind = p_mutation_ind
        self.p_mutation_gen = p_mutation_gen
        self.k_point = k_point

        self.population = self.rng.integers(low=0, high=self.count_engineer, size=(self.population_size, self.count_tasks))

    def fitness(self):
        ret_score = []
        for individual in self.population:
            temp_score = []
            k = 0
            helper_sum = {}
            temp_sum = 0
            # for i, enj in enumerate(individual):
            #     if k <= self.count_engineer:
            #         if enj in helper_sum:
            #             helper_sum[enj] += self.time_array[i] * self.coeff_array[enj][self.tasks[i] - 1]
            #         else:
            #             helper_sum[enj] = self.time_array[i] * self.coeff_array[enj][self.tasks[i] - 1]
            #         k += 1
            #     if k == self.count_engineer or i == len(individual) - 1:
            #         temp_score.append(helper_sum[max(helper_sum, key=helper_sum.get)])
            #         k = 0
            #         helper_sum.clear()
            #         temp_sum = 0
            # ret_score.append(sum(temp_score))
            for i, enj in enumerate(individual):
                if enj in helper_sum:
                    helper_sum[enj] += self.time_array[i] * self.coeff_array[enj][self.tasks[i] - 1]
                else:
                    helper_sum[enj] = self.time_array[i] * self.coeff_array[enj][self.tasks[i] - 1]
            ret_score.append(helper_sum[max(helper_sum, key=helper_sum.get)])


        return ret_score

    def selection(self):
        self.selected_population = self.population[np.argsort(self.fitness())][:self.size_selection]
        return self.selected_population

    def crossover(self):
        n_children = self.population_size - self.size_selection
        idx1 = self.rng.integers(low=0, high=self.size_selection, size=n_children)
        shift = self.rng.integers(low=1, high=self.size_selection - 1, size=n_children)
        idx2 = (idx1 + shift) % self.size_selection

        slice_point = self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)
        self.new_generation = np.where(
            np.arange(self.count_tasks)[np.newaxis] < slice_point[:, np.newaxis], # (1, max_len_individual) < (n_children, 1) => (n_children, max_len_individual)
            self.selected_population[idx1],
            self.selected_population[idx2]
        )
        # slice_points = []
        # for i in range(self.k_point):
        #     a = self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)
        #     slice_points.append(np.array(self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)))
        # slice_points = np.array(slice_points)
        # slice_points.sort(axis=-1)
        # result_mask = []
        # slice_point1 = self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)
        # for i, slice_point in enumerate(slice_points):
        #     if i == 0:
        #         result_mask = np.arange(self.count_tasks)[np.newaxis] <= slice_point[:, np.newaxis]
        #     else:
        #         if i % 2 == 1:
        #             result_mask = np.logical_or(result_mask, np.arange(self.count_tasks)[np.newaxis] > slice_point[:, np.newaxis])
        #         else:
        #             result_mask = np.logical_or(result_mask, np.arange(self.count_tasks)[np.newaxis] <= slice_point[:, np.newaxis])
        # slice_point1 = self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)
        # slice_point2 = self.rng.integers(low=1, high=self.count_tasks - 1, size=n_children)
        # for i in range(slice_point1.size):
        #     slice_point1[i] = min(slice_point1[i], slice_point2[i])
        #     slice_point2[i] = max(slice_point1[i], slice_point2[i])
        #
        # self.new_generation = np.where(
        #     result_mask,
        #     self.selected_population[idx1],
        #     self.selected_population[idx2]
        # )

        return self.new_generation

    def mutation(self):
        mask = self.rng.random(self.new_generation.shape[0]) <= self.p_mutation_ind
        mutation_children = self.new_generation[mask]
        self.new_generation[mask] = np.where(
            self.rng.random(size=mutation_children.shape) <= self.p_mutation_gen,
            self.rng.integers(low=0, high=self.count_engineer, size=mutation_children.shape),
            mutation_children
        )
        return self.new_generation

    def step(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.population = np.concatenate([self.selected_population, self.new_generation], axis=0)





peoples_coeff = []
fin = open("input.txt", "r")
n = int(fin.readline())
tasks = np.array(list((map(int, fin.readline().split()))))
times = np.array(list(map(float, fin.readline().split())))
m = int(fin.readline())
for i in range(m):
    peoples_coeff.append(list(map(float, fin.readline().split())))
#print(peoples_coeff)

ga = Generic_algorithm(tasks, n, m, 150, times, peoples_coeff, 40, 0.3, 0.2, 100)
for i in range(2000):
    print(i)
    ga.step()
print(ga.population[0] + 1)
