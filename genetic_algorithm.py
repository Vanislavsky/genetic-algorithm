import numpy as np


class Generic_algorithm:
    def __init__(self,
                 *,
                 tasks_levels: np.ndarray,
                 count_tasks: int,
                 count_engineer: int,
                 population_size: int,
                 time_array: np.ndarray,
                 coeff_array: np.ndarray,
                 selection_size: int,
                 p_mutation_ind: float,
                 p_mutation_gen: float,
                 k_point: int = 1,
                 fitness_by_total_time: bool = False):
        self.tasks_levels = tasks_levels
        self.count_tasks = count_tasks
        self.count_engineer = count_engineer
        self.population_size = population_size
        self.time_array = time_array
        self.coeff_array = coeff_array
        self.selection_size = selection_size
        self.p_mutation_ind = p_mutation_ind
        self.p_mutation_gen = p_mutation_gen
        self.k_point = k_point
        self.fitness_by_total_time = fitness_by_total_time

        self._rng = np.random.default_rng()
        self._population = self._rng.integers(low=0, high=self.count_engineer, size=(self.population_size, self.count_tasks))

    def fitness_max_time_by_engineer(self):
        ret_score = []
        for individual in self._population:
            enginer_tasks_sums = {}
            for i, enj in enumerate(individual):
                if enj in enginer_tasks_sums:
                    enginer_tasks_sums[enj] += self.time_array[i] * self.coeff_array[enj][self.tasks_levels[i] - 1]
                else:
                    enginer_tasks_sums[enj] = self.time_array[i] * self.coeff_array[enj][self.tasks_levels[i] - 1]
            ret_score.append(enginer_tasks_sums[max(enginer_tasks_sums, key=enginer_tasks_sums.get)])

        return ret_score

    def fintess_total_time(self):
        ret_score = []
        for individual in self._population:
            temp_score = []
            k = 0
            enginer_tasks_sums = {}
            for i, enj in enumerate(individual):
                if k <= self.count_engineer:
                    if enj in enginer_tasks_sums:
                        enginer_tasks_sums[enj] += self.time_array[i] * self.coeff_array[enj][self.tasks_levels[i] - 1]
                    else:
                        enginer_tasks_sums[enj] = self.time_array[i] * self.coeff_array[enj][self.tasks_levels[i] - 1]
                    k += 1
                if k == self.count_engineer or i == len(individual) - 1:
                    temp_score.append(enginer_tasks_sums[max(enginer_tasks_sums, key=enginer_tasks_sums.get)])
                    k = 0
                    enginer_tasks_sums.clear()
            ret_score.append(sum(temp_score))

        return ret_score

    def fitness(self):
        if self.fitness_by_total_time:
            return self.fintess_total_time()
        else:
            return self.fitness_max_time_by_engineer()

    def selection(self):
        self._selected_population = self._population[np.argsort(self.fitness())][:self.selection_size]
        return self._selected_population

    def crossover(self):
        n_children = self.population_size - self.selection_size
        idx1 = self._rng.integers(low=0, high=self.selection_size, size=n_children)
        shift = self._rng.integers(low=1, high=self.selection_size - 1, size=n_children)
        idx2 = (idx1 + shift) % self.selection_size

        slice_points = []
        arr = np.arange(1, self.count_tasks - 1)
        for i in range(n_children):
            points = self._rng.choice(arr, self.k_point, replace=False)
            slice_points.append(points)
        slice_points = np.array(slice_points).T
        slice_points.sort(axis=0)

        mask = [False for i in range(self.count_tasks)]
        all_points = np.arange(self.count_tasks)[np.newaxis]

        for i, slice_point in enumerate(slice_points):
            if i % 2 == 1:
                continue
            if i == 0:
                mask = np.logical_or(mask, np.logical_and(all_points >= 0, all_points < slice_points[i][:, np.newaxis]))
            elif i == slice_points.shape[0] - 1:
                print(slice_points[i][:, np.newaxis])
                mask = np.logical_or(mask, np.logical_and(all_points > slice_points[i][:, np.newaxis], all_points < self.count_tasks))
            else:
                mask = np.logical_or(mask, np.logical_and(all_points > slice_points[i][:, np.newaxis],  all_points < slice_points[i+1][:, np.newaxis]))

        self._new_generation = np.where(
            mask,
            self._selected_population[idx1],
            self._selected_population[idx2]
        )

        return self._new_generation

    def mutation(self):
        mask = self._rng.random(self._new_generation.shape[0]) <= self.p_mutation_ind
        mutation_children = self._new_generation[mask]
        self._new_generation[mask] = np.where(
            self._rng.random(size=mutation_children.shape) <= self.p_mutation_gen,
            self._rng.integers(low=0, high=self.count_engineer, size=mutation_children.shape),
            mutation_children
        )
        return self._new_generation

    def step(self):
        self.selection()
        self.crossover()
        self.mutation()
        self._population = np.concatenate([self._selected_population, self._new_generation], axis=0)

    def best_individuals(self, count):
        return self._population[:count] + 1