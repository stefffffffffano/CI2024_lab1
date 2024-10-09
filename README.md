# CI2024_lab1
## Set-cover problem

In order to face the problem, the simplest solution proposed at the beginning was to use an Hill Climber based on a single mutation tweak function. This solution, however, required too many iterations to reach only an acceptable result, that is translated in a large execution time for the last 3 instances of the problem, those that has a UNIVERSE_SIZE = 100_000. Moreover, this solution was not able to converge to a valid solution if the starting point was not valid.
The first thing we have to focus on regarding the proposed solution for this lab, is the definition of the fitness, useful to evaluate the goodness of the phenotype. This is the proposed version:
```python
def fitness(solution: np.ndarray):
    """Returns a tuple containing first the number of elements covered by the solution and then the cost    of the solution (negative->to be maximized)"""
    return (np.sum(np.logical_or.reduce(SETS[solution])), -cost(solution))
```
This function returns a tuple with the number of covered elements in the universe and minus cost, which is used in order to minimize it. The first version of this function returned, as first element of the tuple, 0 or 1 based on the fact that the solution was valid or not. However, in order to have a 'smoother' representation of the goodness of the solution, I introduced this different fitness function.
Intuitively, in the first case, are invalid solutions were treated in the same way, while, in the proposed solution, there are invalid solutions that are more desirable than other invalid solutions, this allows us to better discriminate. 
Moreover, a lot of work has been done in order to evaluate a smarter starting point for the algorithm instead of a randomly genereted solution (or the full one).
There are different strategies I tried: first, I ordered sets based on the cost and I selected them as taken until complite coverage was reached. As an improvement, I tried to create a set of 'critical sets'. This was basically used to set to true those sets that were the only one covering a given element. However, the previous functions have been discarded from the proposed solution, the critical set was computationally expensive and, moreover, there were not critical sets in the simulations, which means that no element, statistically, ended up being covered by one single element.
The only implementation of smarter solution I left in the notebook is the greedy approach, even if it is completely useless for the scope of the course and of the lab, thus, it is left just as a <u>demonstration that lower costs can be reached, but then, my algorithm couldn't improve it</u>.
Indeed, sometimes starting from a very good solution is not the best idea: you could get stuck in a local optima not being able to improve anymore, and this was the case (at least for me).
More information about the implementation of the greedy algorithm are left in the code, since, as specified before, it is marginal. As tweak function, to generate neighbours, I used: 
```python
def multiple_mutation_strength(solution: np.ndarray, strength: float)->np.ndarray:  
    new_solution = solution.copy()
    mask = rng.random(NUM_SETS) < strength
    new_solution = np.logical_xor(solution,mask)
    return new_solution
```
The main idea behind this tweak function is to mutate strength % (on average) elements of the genotype. This way, I was able to change the strength (sometimes called temperature) based on other factors. 
Let's move now to the algorithm proposed: 
the starting point is the empty solution. There are two cicles: the outer one where the tweak uses an higher strength to favor exploration and an inner one, where the strength is lower, to favor exploitation.
The strength is modified based on the convergence of the solution: when we are close to all the elements covered, it is decreased not to add too many sets but still reach a valid solution. When we are far from a valid solution, we increase it, so that, statistically, we are able to tweak more elements and reach an higher coverage (particularly at the beginning). In case the universe is completely covered, we have iterations left to improve the current solution: so we try to increase or decrease the temperature to favor exploration/exploitation based on the previous results. Here is the code: 
```python
    best = np.full(NUM_SETS,False) #Completely empty solution -> no set selected
    count_iteration = 0
    buffer = []
    BUF_LENGTH = int(number_iteration/5)
    solution_fitness = (0,-0)
    for steps_perturbation in range(int(number_iteration_exploration)):
        if(solution_fitness[0]<UNIVERSE_SIZE/1.5):
            strength*=(1+(1-DENSITY)*5)
        elif(solution_fitness[0]>UNIVERSE_SIZE/1.5 and solution_fitness[0]<UNIVERSE_SIZE):
            if(strength>1*DENSITY):
                strength*=1/(DENSITY*15)
        else:
            buffer = buffer[BUF_LENGTH:]
            if(np.sum(buffer)>1):
                strength *= (1+DENSITY)
            else:
                strength *= (1-DENSITY)
        solution_perturbated = multiple_mutation_strength(best,strength*4) #a huge mutation->exploration
        for steps_local in range(number_iteration_exploitation):
            count_iteration+=1
            new_solution = multiple_mutation_strength(solution_perturbated,strength) 
            fitness_new = fitness(new_solution)
            buffer.append(fitness_new > solution_fitness)
            if fitness_new > solution_fitness:
                solution_fitness = fitness_new
                solution_perturbated = new_solution
                best = new_solution.copy()
    ic(count_iteration)
    ic(fitness(best))
    ic(valid(best))
```
Last but not least, parameters that influence the problem have been adapted based on the instance of the problem, which means, Universe_size,set_size and density. As a starting point, I reasoned about the fact that an instance with higher density needs an higher strength to effectively change the solution. At the same time, regarding iterations, I tried to balance them based on the fact that, if the search space is bigger, obviously more iterations are needed but still taking into account that I didn't want it to require too much time for execution. All other changes, as suggested in lab, have been modified based on a trial and error approach: record the result you obtained with many values and try to change them in a smart way accordingly -> repeat. This is the code: 
```python
    BASE_ITERATIONS = 20
    number_iteration = int(BASE_ITERATIONS*math.log(UNIVERSE_SIZE))/2
    number_iteration_exploration = int(number_iteration*0.2)
    number_iteration_exploitation = int(number_iteration - number_iteration_exploration)
    ic(number_iteration, number_iteration_exploration, number_iteration_exploitation)

    #At the beginning I should favor exploration, then I should favor exploitation
    #The strength depends on the density: if the density is high, the solution is less sensitive to small changes
    BASE_STRENGTH = (0.1/UNIVERSE_SIZE)
    strength = BASE_STRENGTH * (1+DENSITY)
```
Also in the code of the proposed algorithm there's an adaptation of the strength based on the density, but it's explained before. 