from deap import tools

#This class extends the DEAP HallOfFame in order to give priority to the YOUNGEST
#winning genotypes. In The Hunting of The Plark, the sparse reward means that all the
#winners have a reward of 1, I therefore think the most recent genotypes will be more
#generalisable and of more interest.
class HallOfFamePriorityYoungest(tools.HallOfFame):

    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if ind.fitness >= self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

"""
This function is just a copy of the function in DEAP but it dumps the best genotype
in the hall of fame every n generations
"""
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, dump_every=None, obs_normalise=None,
                     domain_params_in_obs=None, dummy_agent=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Dump best agent
        if dump_every is not None:
            if gen % dump_every == 0:
                dummy_agent.set_weights(halloffame[0])
                dummy_agent.save_agent(obs_normalise=obs_normalise,
                                       domain_params_in_obs=domain_params_in_obs,
                                       file_name_suffix="_gen_" + str(gen))

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
