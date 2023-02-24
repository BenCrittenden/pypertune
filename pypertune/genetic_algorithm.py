import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from joblib import Parallel, delayed
from tqdm import tqdm_notebook


class GeneticAlgorithm:
    def __init__(
        self,
        env,
        death_rate=0.5,
        pairing_style="softmax",
        n_gene_swaps=0.5,
        blending_style="random",
        mutation_rate=0.1,
    ):
        self.env = env
        self.death_rate = death_rate
        self.pairing_style = pairing_style
        self.n_gene_swaps = n_gene_swaps
        self.blending_style = blending_style
        self.mutation_rate = mutation_rate

    def generate_initial_chromosomes(self, num_chromosomes, random=False):
        """Creates the initial set of chromosomes where a single chromosome is
        a list of all of the parameter values. The initial set is an evenly
        spaced grid over the parameter space.

        Parameters:
        -----------
        num_chromosomes: int
            the maximum number of chromosomes to start off with.

        Returns:
        --------
        chromosome_matrix: np.array
            an array with chromosomes as rows and parameters as columns.
        """

        mins = list(self.env.param_mins.values())
        maxs = list(self.env.param_maxs.values())

        if random:
            chromosomes = []
            for n in range(num_chromosomes):
                chromosomes += [np.random.uniform(mins, maxs)]

            chromosomes = np.array(chromosomes).astype(int)

        else:
            n_points_per_param = int(num_chromosomes ** (1 / len(mins)))

            grid_values = []
            for p in range(len(self.env.param_mins)):
                grid_values += [
                    np.linspace(mins[p], maxs[p], n_points_per_param)
                ]

            chromosomes = np.array(
                list(itertools.product(*grid_values))
            ).astype(int)

        return chromosomes

    def fit_model(self, chromosome):
        """Fits the model using the hyperparameters specified in the chromosome.

        Parameters
        ----------
        chromosome: np.array
            a vector of parameter values

        Returns
        -------
        performance dict: dict
            A dictionary with the hyperparameters used to fit the model and the
            subsequent peformance of those hyperparameters.
        """

        param_keys = self.env.parameters.keys()
        param_dict = {}
        for i, key in enumerate(param_keys):
            param_dict[key] = chromosome[i]

        performance = self.env.test_model(param_dict)

        return {"performance": performance, "parameters": param_dict}

    def survival_of_fittest(self, model_performance):
        """Ranks the chromosomes by their performance and then removes the bottom x%
        as specified by death_rate

        Parameters
        ----------
        model_performance: list
            A list of dicts where each dict has a peformance key, the value of
            which indicates the performance on the last iteration

        Returns
        -------
        survivors: list
            A list of dicts in the same form as the input parameter, but ordered
            according to performance with the worst performers removed.
        """

        performances = [p["performance"] for p in model_performance]

        sorted_args = [
            x
            for x, y in sorted(
                enumerate(performances), key=lambda x: x[1], reverse=True
            )
        ]

        sorted_performances = [model_performance[i] for i in sorted_args]

        cuttoff = len(sorted_performances) - int(
            len(sorted_performances) * self.death_rate
        )

        survivors = sorted_performances[:cuttoff]

        return survivors

    def mingle(self, survivors):
        """Chromosomes are paired up for subsequent breeding.

        Currently this is done randomly so that a single chromosome can be
        in multiple pairings (but cannot pair with itself). The probability
        of a chromosome being used is determined through softmax of it's prior
        peformance over all prior performances.

        It may be desirable to have a parameter that ensures a chromosome is
        only used once, and another which chooses to use/not use softmax.

        Parameters
        ----------
        survivors: list
            a list of dicts where there is a performance key-value pair.

        Returns
        -------
        partners: list
            a list of lists where each sublist has 2 dicts, each taken from
            survivors chromosome. These 2 dicts each contain a single chromosome
            and together they make up the pair that will be mated.
        """

        performances = [s["performance"] for s in survivors]

        probabilities = self.softmax(performances)

        partners = []
        for x in range(len(survivors) // 2):
            partners += [
                np.random.choice(survivors, 2, p=probabilities, replace=False)
            ]

        return partners

    def softmax(self, performances):
        """Straightforward softmax function

        Parameters
        ----------
        performances: list
            a list of floats that represent the performance from the current
            generation

        Returns
        -------
        probabilities: list
            a list of probabilities
        """

        z_exp = [math.exp(i) for i in performances]
        sum_z_exp = sum(z_exp)
        probabilities = [i / sum_z_exp for i in z_exp]

        return probabilities

    def mate(self, partners):
        """Mixes the two chromosomes

        Parameters
        ----------
        partners: list
            List of list of dicts

        Returns
        -------
        offspring: list
            a list of the new chromosomes to be added to the next generation
        """

        offspring = []
        for couple in partners:
            chromosome_female = list(couple[0]["parameters"].values())
            chromosome_male = list(couple[1]["parameters"].values())

            offspring += [self.gene_swap([chromosome_female, chromosome_male])]

        offspring = [child for children in offspring for child in children]

        return offspring

    def gene_swap(self, chromosome_pair):
        """Swap the genes between the chromosomes

        NOTE: Typically in a Genetic Algorithm at this stage you apply a cross-
        over. Here we do a gene-swap, rather than a true crossover.
        Cross-over will mean that certain genes will be swapped in a way that
        is correlated to their position in the chromosome. This may be desirable
        in nature, or AI, but the order of parameters here is arbitrary, so I feel that
        which genes get swapped should be random

        Parameters
        ----------
        chromosome_pair: list
            A list of 2 lists where each sublist is a chromosome. You can think
            of each of these as female and male parent respectively.

        Returns
        -------
        chromosome_pair: list
            A list of 2 lists where each sublist is a chromosome. You can think of
            these as the children of the parents.
        """

        switch_genes = list(range(len(chromosome_pair[0])))

        if type(self.n_gene_swaps) is not int:
            n_swaps = int(len(switch_genes) * self.n_gene_swaps)

        switch_genes = np.random.choice(switch_genes, n_swaps, replace=False)

        for sg in switch_genes:
            female_gene = chromosome_pair[0][sg]
            male_gene = chromosome_pair[1][sg]

            blended_genes = self.blending([female_gene, male_gene], sg)

            chromosome_pair[0][sg] = blended_genes[0]
            chromosome_pair[1][sg] = blended_genes[1]

        return chromosome_pair

    def blending(self, gene_pairs, sg):
        """This is required when genes are (effectively) continuous as in this case.
        Without blending, only the initial genes (parameter values) can be passed
        on. Blending allows new values to be generated.

        Parameters
        ----------
        gene_pairs: list
            a list of two numbers, i.e. two genes, one from each parent.
        sg: int
            the index of the gene along the chromosome

        Returns
        -------
        blended_genes: list
            a list of two numbers, i.e. two genes, a recombination of their parents

        """

        if self.blending_style == "random":
            beta = np.random.random()
            new_female = (beta * gene_pairs[0]) + ((1 - beta) * gene_pairs[1])
            new_male = (beta * gene_pairs[1]) + ((1 - beta) * gene_pairs[0])

        elif self.blending_style == "expanded_crossover":
            viable = False
            while not viable:
                new_children = []
                beta = np.random.random()

                new_children += [
                    (beta * gene_pairs[0]) + ((1 - beta) * gene_pairs[1])
                ]
                new_children += [
                    (beta * gene_pairs[1]) + ((1 - beta) * gene_pairs[0])
                ]
                new_children += [
                    ((1 + beta) * gene_pairs[0]) - ((1 - beta) * gene_pairs[1])
                ]
                new_children += [
                    -((1 - beta) * gene_pairs[0])
                    + ((1 + beta) * gene_pairs[1])
                ]

                valid = []
                mins = list(self.env.param_mins.values())
                maxs = list(self.env.param_maxs.values())
                for child in new_children:
                    if (child < mins[sg]) | (child > maxs[sg]):
                        valid += [False]
                    else:
                        valid += [True]

                if sum(valid) == 2:
                    new_children = list(
                        itertools.compress(new_children, valid)
                    )
                    viable = True
                elif sum(valid) > 2:
                    pick_two_of = list(itertools.compress(new_children, valid))
                    new_children = np.random.choice(
                        pick_two_of, 2, replace=False
                    )
                    viable = True

            new_female = new_children[0]
            new_male = new_children[1]

        blended_genes = [new_female, new_male]

        return blended_genes

    def mutate(self, offspring):
        """adds random mutation. Essentially randomly selects if a gene will
        be mutated and then changes the value to a random value within the
        bounds set by that parameter.

        Parameters
        ----------
        offspring: list
            a list of all the chromosomes that form the next generation

        Returns
        -------
        mutants: list
            a list of chromosomes, where some of the

        """

        mutants = []
        for child in offspring:
            mutate_genes = np.random.random(len(child)) < self.mutation_rate

            for gene in range(len(mutate_genes)):
                if mutate_genes[gene]:
                    min_val = list(self.env.param_mins.values())[gene]
                    max_val = list(self.env.param_maxs.values())[gene]
                    mutant_gene = np.random.uniform(min_val, max_val)

                    child[gene] = mutant_gene

            mutants += [child]

        mutants = self.ensure_correct_dtype(mutants)

        return mutants

    def ensure_correct_dtype(self, mutants):
        """ensures that the data type of the next generation is of the correct
        type, which is essential for some parameters in the model.

        Parameters
        ----------
        mutants: list
            a list of lists

        Returns
        -------
        mutants: np.array
            an object np array where each column (genes) is of the correct type.

        """

        dtypes = [dtype["type"] for dtype in self.env.parameters.values()]

        mutants = np.array(mutants, dtype=object)

        for col in range(mutants.shape[1]):
            if dtypes[col] == "int":
                mutants[:, col] = np.round(
                    mutants.astype(float)[:, col]
                ).astype(int)

        return mutants

    def next_generation(self, survivors, mutants):
        """combines the chromosomes of the survivors and the mutants into a single
        array that make up the next generation

        Parameters
        ----------
        survivors: list
            a list of dicts where the parameters key-value has the chromosomes
        mutants: np.array
            an array with chromosomes as rows and genes (parameters) as columns

        Returns
        -------
        chromosomes_matrix: np.array
            an array with chromosomes as rows and genes (parameters) as columns
        """

        survivor_chromosomes = np.array(
            [list(s["parameters"].values()) for s in survivors]
        )

        chromosomes_matrix = np.append(survivor_chromosomes, mutants, axis=0)

        return chromosomes_matrix

    def optimal_hyperparameters(self):
        """Find the optimal hyperparameters - the best peforming model over all
        generations
        """

        all_performances = self.extract_all_performances()

        max_coords = np.unravel_index(
            all_performances.argmax(), all_performances.shape
        )

        self.optimal_model = self.generational_model_performance[
            max_coords[0]
        ][max_coords[1]]

        return self.optimal_model

    def extract_all_performances(self):
        """Get the performance of all models across all generations

        Returns
        -------
        all_performances: np.array
            an array with the performance from all chromosomes across
            all generations.
        """

        all_performances = []
        for generation in self.generational_model_performance:
            all_chromosomes = []
            for chromosome in generation:
                all_chromosomes += [chromosome["performance"]]

            all_performances += [all_chromosomes]

        # all_performances = np.array(all_performances)
        all_performances = np.array(
            list(itertools.zip_longest(*all_performances, fillvalue=np.nan))
        ).T

        return all_performances

    def print_optimal_hyperparameters(self):
        """Prints the optimal parameters and their values to the screen"""

        optimal_model = self.optimal_model["parameters"]
        print(
            "".join(
                ["Optimal hyperparameters >>> |"]
                + [f" {key}: {optimal_model[key]} |" for key in optimal_model]
            )
        )

    def optimal_classifier(self):
        """Returns the model with parameter values set at the identified optimum

        Returns
        -------
        clf: model object
            model with parameters set to optimum values
        """

        optimal_params = self.optimal_model["parameters"]
        clf = self.env.gen_classifier(optimal_params)
        clf.fit(self.env.x_data, self.env.y_data)

        return clf

    def generational_optimal_models(self):
        """Identifies the best model from each generation

        Returns
        -------
        generational_optimal_models: list
            a list of the best models from each generation
        """

        all_performances = self.extract_all_performances()

        generational_best_model = np.argmax(all_performances, axis=1)

        generational_optimal_models = []
        for generation in range(len(generational_best_model)):
            generational_optimal_models += [
                self.generational_model_performance[generation][
                    generational_best_model[generation]
                ]
            ]

        return generational_optimal_models

    def report(self):
        """Prints the figures that describe peformance over generations"""

        all_performances = self.extract_all_performances()

        generation_performance = []
        for generation in range(all_performances.shape[0]):
            generation_performance += list(
                zip(
                    [generation] * all_performances.shape[1],
                    all_performances[generation, :],
                )
            )

        generation_performance = np.array(generation_performance)

        fig1 = plt.figure(figsize=(12, 12))

        ax1 = fig1.add_subplot(111)
        performance_plot = plt.plot(
            generation_performance[:, 0],
            generation_performance[:, 1],
            ".",
            label="Performance",
        )
        plt.xlabel("Generation")

        fig1.tight_layout()
        plt.show()

        optimal_models = self.generational_optimal_models()

        fig2 = plt.figure(figsize=(12, 8))

        n_params = len(self.env.params)

        for i in range(len(self.env.params.keys())):
            ax = fig2.add_subplot(n_params, 2, 1 + i)
            ax.plot(
                np.arange(len(optimal_models)),
                np.array(
                    [
                        generation["parameters"][
                            list(self.env.params.keys())[i]
                        ]
                        for generation in optimal_models
                    ]
                ),
            )
            ax.set_title(list(self.env.params.keys())[i])
            ax.set_ylim(
                [
                    list(self.env.param_mins.values())[i] - 1,
                    list(self.env.param_maxs.values())[i] + 1,
                ]
            )
            ax.set_xlabel("Episode")
        plt.tight_layout()
        fig2 = plt.gcf()

    def learn(
        self,
        num_chromosomes=100,
        num_generations=6,
        method="serial",
        chromosomes_matrix=[],
        random_init=False,
    ):
        """Main method that coordinates the learning process.

        Parameters
        ----------
        num_chromosomes: int
            upper bound on the number of chromosomes to start off with in the
            first generation.
        num_generations: int
            number of generations to run the genetic algorithm over
        method: str
            "parallel" or "serial". Whether the script fits the chromosomes within
            each generation in parallel or in serial.
        chromosome_matrix: np.array
            optional parameter where you can specify the starting group of chromosomes
            instead of it being automatically generated.

        Returns
        -------
        model object
            model with parameters set to optimum values

        """

        if chromosomes_matrix == []:
            chromosomes_matrix = self.generate_initial_chromosomes(
                num_chromosomes, random=random_init
            )

        self.generational_model_performance = []
        for g in tqdm_notebook(range(num_generations)):
            if method == "serial":
                model_performance = []
                for i in range(chromosomes_matrix.shape[0]):
                    model_performance += [
                        self.fit_model(chromosomes_matrix[i, :])
                    ]

            elif method == "parallel":
                n_cores = int(os.getenv("NUM_CPUS", os.cpu_count()))

                model_performance = Parallel(
                    n_jobs=n_cores,
                    batch_size=chromosomes_matrix.shape[0] // n_cores,
                    verbose=40,
                )(
                    delayed(self.fit_model)(chromosome)
                    for chromosome in chromosomes_matrix
                )

            else:
                print(
                    "Error: method arg must be either 'parallel' or 'serial'"
                )

            self.generational_model_performance += [model_performance]

            survivors = self.survival_of_fittest(model_performance)

            partners = self.mingle(survivors)

            offspring = self.mate(partners)

            mutants = self.mutate(offspring)

            chromosomes_matrix = self.next_generation(survivors, mutants)

        self.optimal_hyperparameters()
        self.print_optimal_hyperparameters()

        return self.optimal_classifier()
