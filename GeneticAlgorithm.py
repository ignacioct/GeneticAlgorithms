from concurrent import futures
import math
import random
import requests

def main():

    # Configuration module
    POP_SIZE = 300
    TOURNAMENT_FACTOR = 0.1
    MUTATION_FACTOR = 0.005
    ELITIST_FACTOR = 0.2
    COMPLETE = True             # If True, we face the 384-bit problem; otherwise we are doing the 64-bit
    CONCURRENT = True           # If True, we use concurrent evaluation to speedup the process; otherwise we use sequential
    ROG = False                  # If True, we apply the ROG crossover. If false, we apply the regular uniform crossover
    TRAINING_CYCLES = 500

    model = GenAlg(POP_SIZE, COMPLETE)
    model.training_loop(TRAINING_CYCLES, TOURNAMENT_FACTOR, MUTATION_FACTOR, ELITIST_FACTOR, COMPLETE, CONCURRENT, ROG)


class GenAlg():
    """
    Implementation of the genetic algorithm to solve the weather station problem
    """

    def __init__(self, pop_size_input, isComplete):
        self.POP_SIZE = pop_size_input              # Testing made with 10, real values around 50 or 100
        if isComplete is True:
            self.CHROM_SIZE = 384
        else:
            self.CHROM_SIZE = 64
        self.population = []                # Empty list init for the population
        self.fitnesses = []                 # Empty list init for the fitness values of the population
        self.TOURNAMENT_FACTOR = 0          # Tournament factor, will be assigned a value in the train function call
        self.MUTATION_FACTOR = 0            # Mutation factor, will be assigned a value in the train function call
        self.ELITIST_FACTOR = 0
        self.evaluations = 0                # Number of http/get requests succesfully made


    def initialization(self):
        """
        Initializates the population randomly
        """

        for _ in range(self.POP_SIZE):
            chromosome = ''                             # Chromosome will be a string to treat binary features correctly
            for _ in range(self.CHROM_SIZE):
                chromosome += str(random.randint(0, 1)) # Random [0;1]
            self.population.append(chromosome)          # Adding each new random chromosome to the population
        return self.population

    def get_request_fitness(self, chromosome, isComplete):
        """
        Fitness Function, to be chosen
        """

        return 1

    def evaluation(self, isComplete):
        """
        Evaluates the whole population of chromosomes
        """

        self.fitnesses = []                         # Empty list init
        for i in range(self.POP_SIZE):
            self.fitnesses.append(self.get_request_fitness(self.population[i], isComplete))

    def concurrent_evaluation(self, isComplete):
        """
        Evaluates the whole population of chromosomes.
        Uses concurrent package to paralelize the process, as it is a bottleneck for the whole process
        """
        self.fitnesses = []     # Empty list init
        with futures.ThreadPoolExecutor(max_workers=50) as execute:
            future = [
                execute.submit(self.get_request_fitness, self.population[j], isComplete)
                for j in range(self.POP_SIZE)]
        self.fitnesses = [f.result() for f in future]

    def tournament(self):
        """
        Selects the best individuals by facing them to each other and keeping the best.
        Applied to the whole population.
        """
        temp_population = []         # Temporal place for the newly-created population
        for _ in range(self.POP_SIZE):

            # Get tournament size as the floored integer of the Population Size * Tournament Percentage (aka factor)
            tournament_size = math.floor(self.TOURNAMENT_FACTOR * self.POP_SIZE)

            # Selects a random fraction of the total population to participate in the tournament
            tournament_selected = random.sample(range(self.POP_SIZE), tournament_size)

            # Choose the fittest
            fittest_index = min([self.fitnesses[index] for index in tournament_selected])   # Getting index of fittest
            fittest = self.fitnesses.index(fittest_index)

            temp_population.append(self.population[fittest])

        self.population = temp_population       # Moving from the temporal place to overwriting the population

    def elitist_tournament(self):
        """
        Selects the best individuals by facing them to each other and keeping the best.
        The best individuals pass directly to the parents pool.
        """

        temp_population = []         # Temporal place for the newly-created population

        ordered_population = [x for _,x in sorted(zip(self.fitnesses, self.population))] # Population ordered by fitness
        elitism_size = math.floor(self.POP_SIZE*self.ELITIST_FACTOR)
        rest_population = self.POP_SIZE - elitism_size

        best_individuals = ordered_population[:elitism_size]

        temp_population = best_individuals

        for _ in range(rest_population):

            # Get tournament size as the floored integer of the Population Size * Tournament Percentage (aka factor)
            tournament_size = math.floor(self.TOURNAMENT_FACTOR * self.POP_SIZE)

            # Selects a random fraction of the total population to participate in the tournament
            tournament_selected = random.sample(range(self.POP_SIZE), tournament_size)

            # Choose the fittest
            fittest_index = min([self.fitnesses[index] for index in tournament_selected])   # Getting index of fittest
            fittest = self.fitnesses.index(fittest_index)

            temp_population.append(self.population[fittest])    # Adding the tournament results to the best individuals

        self.population = temp_population

    def crossover(self):
        """
        Applies the crossover operator, combinating the info of two chromosomes into one.
        """
        temp_population = []        # Temporal place for the newly-created population

        # Greater loop, traversing all chromosomes in the population, 2 by 2
        for i in range(0, self.POP_SIZE, 2):
            child1, child2 = '', ''    # Empty intialization of children

            # Inner loop, traversing the content of each chromosome
            for j in range(self.CHROM_SIZE):
                # Not combining both randoms to preserve variability
                if random.randint(0, 1) == 1:
                    child1 += self.population[i][j]
                else:
                    child1 += self.population[i + 1][j]
                if random.randint(0, 1) == 1:
                    child2 += self.population[i][j]
                else:
                    child2 += self.population[i + 1][j]

            temp_population.append(child1)
            temp_population.append(child2)

        self.population = temp_population       # Moving from the temporal place to overwriting the population

    def ROG_crossover(self):
        """
        Random Offspring Generator, proposed by Rocha & Neves.
        If both fathers have the same genotype, one of the children will be random
        """

        temp_population = []        # Temporal place for the newly-created population

        # Greater loop, traversing all chromosomes in the population, 2 by 2
        for i in range(0, self.POP_SIZE, 2):
            child1, child2 = '', ''    # Empty intialization of children

            # Checking if both parents are equal
            if self.population[i] == self.population[i+1]:
                #Generating a random offspring
                for _ in range(self.CHROM_SIZE):
                    child1 += str(random.randint(0, 1)) # Random [0;1]
                
                # Inner loop, creating a regular descendant
                for j in range(self.CHROM_SIZE):
                    # Not combining both randoms to preserve variability
                    if random.randint(0, 1) == 1:
                        child2 += self.population[i][j]
                    else:
                        child2 += self.population[i + 1][j]
                
            else:
                # Inner loop, traversing the content of each chromosome
                for j in range(self.CHROM_SIZE):
                    # Not combining both randoms to preserve variability
                    if random.randint(0, 1) == 1:
                        child1 += self.population[i][j]
                    else:
                        child1 += self.population[i + 1][j]
                    if random.randint(0, 1) == 1:
                        child2 += self.population[i][j]
                    else:
                        child2 += self.population[i + 1][j]

            temp_population.append(child1)
            temp_population.append(child2)

        self.population = temp_population       # Moving from the temporal place to overwriting the population


    def mutation(self):
        """
        Applies the mutation operator, changing the content of a chromosome given a mutation factor
        """
        temp_population = []        # Temporal place for the newly-created population
        # Greater loop, traversing all chromosomes in the population
        for i in range(self.POP_SIZE):
            mutated = ''            # Empty intialization of mutated result
            # Inner loop, traversing the content of each chromosome
            for j in range(self.CHROM_SIZE):
                if random.random() <= self.MUTATION_FACTOR :
                    # x = 1-x turns 1->0 and 0->1
                    mutated += str((1 - int(self.population[i][j])))
                else:
                    mutated += self.population[i][j]
            temp_population.append(mutated)
        self.population = temp_population       # Moving from the temporal place to overwriting the population

    def training_loop(self, training_cycles, tournament_factor, mutation_factor, elitist_factor, isComplete, concurrency, rog_crossover):
        """
        Training loop, which consist on evaluation, selection of the fittest, crossover and mutation
        """
        # Declaration of extra global variables
        self.TOURNAMENT_FACTOR = tournament_factor
        self.MUTATION_FACTOR = mutation_factor
        self.ELITIST_FACTOR = elitist_factor

        self.initialization()

        for i in range(training_cycles):

            # Controlling the two types of evaluations
            if concurrency is True:
                self.concurrent_evaluation(isComplete)
            else:
                self.evaluation(isComplete)

            # Getting the fittest individual of a generation, its index and its fitness value
            fittest_value = min(self.fitnesses)
            fittest_chromosome_index = self.fitnesses.index(fittest_value)
            fittest_chromosome = self.population[fittest_chromosome_index]

            # Printing information
            print("Generation: " + str(i) + "\tBest score: " + str(fittest_value) +
                "\tBest chromosome: " + fittest_chromosome)
                

            # Tournament, crossover and mutation
            if self.ELITIST_FACTOR > 0:
                self.elitist_tournament()
            else:
                self.tournament()
            
            if rog_crossover is True:
                self.ROG_crossover()
            else:
                self.crossover()

            self.mutation()

            # Early stopping criteria
            # For this case in which we know we are going to hit 0, it should be enough coding reach-zero stop criteria
            if fittest_value == 0:
                print("[i] Zero reached, stop criteria applied")
                print("Zero-reached generation: " + str(i) + "\tScore: " + str(fittest_value) +
                      "\tChromosome: " + fittest_chromosome)
                break




if __name__ == "__main__":
    main()
