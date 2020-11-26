import copy
import math
import operator
import random
import sys
from concurrent import futures

import numpy as np
import requests


class FitnessFunctionCaller:
    """Class for returning the fitness function of an individual."""

    def __init__(self, *args):

        functional_parts = []
        # Full case with 10 motors
        if len(args) > 0:
            for arg in args:
                functional_parts.append(arg)


    def call(self) -> float:
        """Returns the fitness function"""

        return 1# Fitness function


class Individual:
    """Candidate solution to the problem. Made by a functional value and a variance."""

    def __init__(self, is10, **kwargs):

        functional = kwargs.get("functional", None)
        variance = kwargs.get("variance", None)
        self.is10 = is10
        if is10 is False:
            self.motorNumber = 4
        else:
            self.motorNumber = 10

        if len(kwargs) == 0:
            self.functional = [
                np.random.uniform(-180, 181) for _ in range(self.motorNumber)
            ]
            self.variance = [
                np.random.uniform(100, 360) for _ in range(self.motorNumber)
            ]
        else:
            self.functional = functional
            self.variance = variance

        self.fitness = sys.float_info.max  # irrational high value

    def update_fitness(self, incoming):
        """Update fitness function"""

        self.fitness = incoming

    def update_variance(self, incoming):
        """Update variance function"""

        for i in range(self.motorNumber):
            self.variance[i] = incoming[i]


class EvolutiveStrategyOneIndividual:
    """Evolution strategy made only one solution with mutation."""

    def __init__(self, c, is10):
        self.population = 1
        self.pool = []
        for _ in range(self.population):  # reusable for bigger populations
            indv = Individual(is10)
            self.pool.append(indv)
        self.successes = []  # 1 if improves, otherwise 0
        self.psi = (
            self.successes.count(1) / 10
        )  # ratio of improvement in the last 10 generations
        self.c = c  # coefficient for 1/5 rule
        self.evaluations = 0
        self.lastFitness = sys.float_info.max  # irrational high value

    def mutation(self):
        """A temporal solution is produced, being the second individual the result of the mutation"""

        # Creating temporal dictionaries
        self.temporalPool = []
        temporal_functional = []
        temporal_variance = []

        for i in range(self.pool[0].motorNumber):
            # Functional mutation
            temporal_functional.append(
                self.pool[0].functional[i]
                + np.random.normal(scale=self.pool[0].variance[i])
            )

        temp_indv = Individual(
            is10=self.pool[0].is10,
            functional=temporal_functional,
            variance=self.pool[0].variance,
        )

        self.temporalPool.append(temp_indv)

    def evaluation(self):
        """Selecting the best of the two individual and evaluating them"""

        # Getting the fitness evaluations of the former individual and the mutated one
        formerIndividualCaller = FitnessFunctionCaller(*(i for i in self.pool[0].functional))
        temporalIndividualCaller = FitnessFunctionCaller(
            *(i for i in self.temporalPool[0].functional)
        )
        formerIndividualFitness = formerIndividualCaller.call()
        temporalIndividualFitness = temporalIndividualCaller.call()

        self.evaluations += 2
        # formerBetter is True if the mutation did not improve the fitness over the father
        if formerIndividualFitness <= temporalIndividualFitness:
            formerBetter = True
        else:
            formerBetter = False

        # bestFitness in between former and temporal
        bestFitness = min(formerIndividualFitness, temporalIndividualFitness)

        # If the child did improved, we change the pool to the temporal pool
        if formerBetter is False:
            self.pool = copy.deepcopy(self.temporalPool)

        # In any case, we delete the temporal pool at this point
        del self.temporalPool

        # Variance mutation
        for i in range(self.pool[0].motorNumber):
            self.pool[0].variance[i] = self.ruleOneFifth(self.pool[0].variance[i])

        # Update fitness function
        self.pool[0].update_fitness(bestFitness)

        # Adding 1 to the success matrix if the best individual is the child
        if formerBetter is True:
            if len(self.successes) < 10:
                self.successes.append(0)
            else:
                self.successes.pop(0)
                self.successes.append(0)
        else:
            if len(self.successes) < 10:
                self.successes.append(1)
            else:
                self.successes.pop(0)
                self.successes.append(1)

        # Updating last fitness
        self.lastFitness = bestFitness

        # Update psi
        self.psi = (
            self.successes.count(1) / 10
        )  # ratio of improvement in the last 10 generations

    def trainingLoop(self, maxCycles):
        """Training loop, controlled at maximum by the last cicle"""

        for cycle in range(maxCycles):
            self.mutation()
            self.evaluation()

            formerResults = []
            if len(formerResults) > 10:
                formerResults.pop(0)
            formerResults.append(
                "Generation: "
                + str(cycle)
                + "\tBest fitness: "
                + str(self.pool[0].fitness)
                + "\nBest chromosome: "
                + str(self.pool[0].functional)
            )
            print(
                "Generation: "
                + str(cycle)
                + "\tBest fitness: "
                + str(self.pool[0].fitness)
                + "\nBest chromosome: "
                + str(self.pool[0].functional)
            )
            stopping = False
            for i in range(len(self.pool[0].functional)):
                if self.pool[0].variance[i] < 0.0001:
                    stopping = True

            if stopping == True:
                print("Early stopping applied")
                print(formerResults[0])
                break

    def ruleOneFifth(self, formerVariance) -> float:
        """Applies the one fifth rule given the former variance"""

        # Update psi
        self.psi = (
            self.successes.count(1) / 10
        )  # ratio of improvement in the last 10 generations

        if self.psi < 0.2:
            return self.c * formerVariance

        elif self.psi > 0.2:
            return self.c / formerVariance

        else:
            return formerVariance


class EvolutiveStrategyMultiple:
    """Evolution strategy made with a population of individuals."""

    def __init__(self, population, family_number, tournament_factor, is10):
        self.population = population
        self.pool = []
        for _ in range(self.population):
            indv = Individual(is10)
            self.pool.append(indv)
        self.family_number = family_number
        self.tau = 1 / math.sqrt(2 * math.sqrt(self.pool[0].motorNumber))
        self.zero_tau = 1 / math.sqrt(2 * self.pool[0].motorNumber)
        self.tournament_factor = tournament_factor
        self.evaluations = 0

    def element_per_list(self, lista):
        """Auxiliar function; given a list of lists, picks a random for each position searching in all lists"""
        temporal_list = []
        for position in range(len(lista[0])):
            rnd = random.randint(0, (self.family_number - 1))
            temporal_list.append(lista[rnd][position])

        return temporal_list

    def tournament(self):
        """
        Selects the best individuals by facing them to each other and keeping the best.
        Returns a population of the best inidividuals
        """

        len_population = self.family_number * self.population
        temp_population = []  # Temporal place for the newly-created population

        for _ in range(len_population):

            # Get tournament size as the floored integer of the Population Size * Tournament Percentage (aka factor)
            tournament_size = math.floor(self.tournament_factor * self.population)

            # Selects a random fraction of the total population to participate in the tournament
            tournament_selected = random.sample(range(self.population), tournament_size)

            # Choose the fittest
            fitnesses = []
            indexes = []
            for index in tournament_selected:
                fitnesses.append(self.pool[index].fitness)
                indexes.append(index)

            fittest_index = indexes[fitnesses.index(min(fitnesses))]

            fittest = self.pool[fittest_index]

            temp_population.append(fittest)

        return temp_population  # Returning the new population

    def crossover(self, pool):
        """Returns a pool of children, given a the pool of individuals of the last generation and a family number."""

        temporal_pool = []

        random.shuffle(pool)  # randomize the pool of individuals, to randomize crossover

        counter = 0  # controls the loops logic

        avg_functionals = [0] * pool[0].motorNumber  # functional list for the newborns (must be restarted with 0-init)
        avg_variances = ([])  # variances list for the newborns (must be restarted by recasting)

        for indv in pool:

            if counter != (self.family_number - 1):  # not the last member of the family
                for position in range(indv.motorNumber):
                    avg_functionals[position] += indv.functional[position]  # adds each functional of the current ind to corresponding positions
                avg_variances.append(indv.variance)  # adds the variance to the list of parent variances
                counter += 1

            else:  # last member of the family -> extra functions
                for position in range(indv.motorNumber):
                    avg_functionals[position] += indv.functional[position]
                    avg_functionals[
                        position
                    ] /= (
                        self.family_number
                    )  # no more sums left, time to divide by family number

                avg_variances.append(indv.variance)

                # Transforming the list of lists to a list of variances, with a random variance of the parents for each position
                avg_variances = self.element_per_list(avg_variances)

                # Adding the individual to the temporal pool
                temp_indv = Individual(
                    is10=pool[0].is10,
                    functional=avg_functionals,
                    variance=avg_variances,
                )
                temporal_pool.append(temp_indv)

                # Restarting variables, as this family has finished
                counter = 0
                avg_functionals = [0] * pool[0].motorNumber
                avg_variances = []

        """
        With this implementation, if population mod family number is not zero, those parents at the end wont create any child.
        To cope with that, the parents pool is shuffled. This should not be a problem, just 1 or 2 will be excluded.
        At the end, we get the same number of children, so the rest of the operators remain unchanged, and convergence will work just fine.
        """

        return temporal_pool

    def mutation(self, pool, scaling):
        """
        Given a pool of individuals, mutates all individuals
            functionals get mutated by a Gaussian distribution
            variances get decreased by a Gaussian scheme
        """

        for individual in pool:
            for i in range(individual.motorNumber):

                # Functional mutation
                individual.functional[i] += np.random.normal(
                    loc=0, scale=individual.variance[i]
                )
                # Variance mutation
                if scaling is True:
                    individual.variance[i] = (
                        individual.variance[i]
                        * np.exp(np.random.normal(loc=0, scale=self.tau))
                        * np.exp(np.random.normal(loc=0, scale=self.zero_tau))
                    )
                else:
                    individual.variance[i] = individual.variance[i] * np.exp(
                        np.random.normal(loc=0, scale=self.tau)
                    )

        return pool

    def concurrent_evaluation(self, pool):
        """Given a pool of individuals, return a list with its fitness functions"""

        callers = []  # list of caller objects of individuals
        for individual in pool:
            individual_caller = FitnessFunctionCaller(*(i for i in individual.functional))
            callers.append(individual_caller)
        with futures.ThreadPoolExecutor(max_workers=50) as execute:
            future = [execute.submit(callers[i].call) for i in range(len(pool))]

        self.evaluations += len(future)

        fitnesses = [f.result() for f in future]  # list of fitness of the pool
        return fitnesses

    def selection(self, children_pool):
        """Given a pool of mutated children, and using self.pool (parent's pool), selects the best individuals"""

        fitnesses = []
        combined_pool = copy.deepcopy(
            self.pool
        )  # introducing parents to a combined pool
        combined_pool.extend(children_pool)  # introducing childs to a combined pool

        for i in range(len(self.pool)):
            fitnesses.append(self.pool[i].fitness)

        fitnesses.extend(
            self.concurrent_evaluation(children_pool)
        )  # list of fitnesses of the combined pool

        for i in range(len(combined_pool)):
            combined_pool[i].fitness = fitnesses[i]

        combined_pool.sort(key=operator.attrgetter("fitness"))

        # ordered_combined_pool = [x for _,x in sorted(zip(fitnesses, combined_pool))]    # Population ordered by fitness

        self.pool = copy.deepcopy(combined_pool[: self.population])  # The pool will now be the best individuals of both parents and children

        fitnesses.sort()
        for i in range(len(self.pool)):
            self.pool[i].fitness = fitnesses[i]

        return

    def training_cycle(self, max_cycles, scaling):
        """Training loop, controlled at maximum by the max cycle"""

        fitnesses = self.concurrent_evaluation(self.pool)

        for i in range(len(self.pool)):
            self.pool[i].fitness = fitnesses[i]

        for cycle in range(max_cycles):

            temp_pool = self.tournament()
            temp_pool = self.crossover(temp_pool)
            temp_pool = self.mutation(temp_pool, scaling)
            self.selection(temp_pool)

            print(
                "Generation: "
                + str(cycle)
                + "\t Evaluation: "
                + str(self.evaluations)
                + "\tBest fitness: "
                + str(self.pool[0].fitness)
                + "\nBest chromosome: "
                + str(self.pool[0].functional)
                + "\n"
                + str(self.pool[0].variance)
            )

            if self.pool[0].fitness == 0.0:
                print("Early stopping applied")

                print(
                    "Generation: "
                    + str(cycle)
                    + "\t Evaluation: "
                    + str(self.evaluations)
                    + "\tBest fitness: "
                    + str(self.pool[0].fitness)
                    + "\nBest chromosome: "
                    + str(self.pool[0].functional)
                    + "\n"
                    + str(self.pool[0].variance)
                )
                break


def main():

    # Code for strategy of 1 individual
    # ee = EvolutiveStrategyOneIndividual(c=ce, is10=True)
    # ee.trainingLoop(10000)

    # Code for strategy with the best results
    ee = EvolutiveStrategyMultiple(
        population=300, family_number=2, tournament_factor=0.05, is10=True
    )
    ee.training_cycle(1000, scaling=True)


if __name__ == "__main__":
    main()
