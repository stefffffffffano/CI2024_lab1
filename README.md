# CI2024_lab1
## Set-cover problem
## to convert a file in PDF-> Ctrl + shift + P-> Digitate Markdown PDF: Export (pdf) e selezionate l'opzione.

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
Costs and number of steps for different instances: 
| Instance | Cost | Number of steps |
| --------- | --------- | --------- |
| UNIVERSE_SIZE = 100, SET_SIZE = 10, DENSITY = 0.2            | 265.9995711684986   | 333   |
| UNIVERSE_SIZE = 1_000, SET_SIZE = 100, DENSITY =  0.2        | 7947.6832760372445  | 728    |
| UNIVERSE_SIZE = 10_000 , SET_SIZE = 1_000 , DENSITY =  0.2   | 170034.64124024883  | 1332    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.1  | 2437831.577057764   | 2116    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.2  | 2637136.1952890246  | 2116    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.3  | 2520546.5410716645 | 2116    |

## Lab 1 review
For this first lab, I was assigned to review the code written by two collegues. I will analyze separately both of them highlighting suggestions I tried to give them. The strategy followed is: try to improve the strategy that they have chosen. So, basically, if they picked Hill Climbing, for example, give suggestions about the tweak or the fitness function in order to improve the result. I didn't suggest different algorithms because I think this is not the point of a review: I should try to build on the work you've provided, not change it completely.
The first solution is by Giorgio Silvestre, who decided to use a random-Mutation Hill Climber with multiple mutation and simulated annealing. This is the code:
```python
solution = rng.random(NUM_SETS) < 0.5
solution_fitness = fitness(solution)
ic(solution_fitness)
history = [solution_fitness[1]]
temp = TEMPERATURE

for step in range(N_STEPS):
    new_solution = multi_tweak(solution)
    history.append(fitness(new_solution)[1])
    if fitness(new_solution) > solution_fitness or (fitness(new_solution)[0] and simulated_annealing(solution_fitness, fitness(new_solution), temp)):
        solution = new_solution
        solution_fitness = fitness(solution)
    temp *= 0.99
ic(fitness(solution))
ic(history.index(fitness(solution)[1]))

plt.figure(figsize=(14, 8))
plt.plot(
    range(len(history)),
    list(accumulate(history, max)),
    color="red",
)
_ = plt.scatter(range(len(history)), history, marker=".")
```
Where the function simulated_annealing is: 
```python
def simulated_annealing(solution_fitness, new_solution_fitness, temp):
    accepting_probability = np.exp(-(solution_fitness[1] - new_solution_fitness[1]) / temp)
    #ic(accepting_probability)
    return np.random.random() < accepting_probability
```
And, since I make a reference to it in the review, I also provide the definition of the fitness function:
```python
def fitness(solution):
    """Returns the fitness of a solution (to be maximized)"""
    return (valid(solution), -cost(solution))
```
This is the comment I provided to Giorgio:  
```
A couple of comments about the simulated annealing that, if I understood correctly, is the 'delivered' algorithm. The algorithm seems to work very well since you got a significant improvement, but I think you could slightly improve it by making some changes.
First of all, I think it could be useful to redefine the fitness function. Currently, your fitness function returns a tuple (valid/invalid, -cost), what if it returned (coverage, cost)? This should lead to better results because all invalid solutions are not considered the same, but they are better depending on the number of elements covered. It's quite simple to implement it with numpy, using np.sum(np.logical_or.reduce(SETS[solution])).
Moreover, I think also that the starting point for the algorithm could be improved. Currently, you are taking a random solution that selects a set with a 50% probability. This seems to be too high for the last instances, which means that you start with a cost that is higher. I tried your algorithm starting with a lower probability and it seems to work better for the last instances. An idea could be to modify parameters (all of them) based on the instance (on their parameters). In particular, you could initialize temperature and starting point of the algorithm.
If you want, also the multiple_tweak() function could be slightly improved by using a strength. In particular, you use an additional parameter that regulates the percentage of elements that are swapped when the tweak function is called. This allows you to introduce one more variable that can depend on the density of the instance. Indeed, an instance with an higher density typically requires a tweak with an higher strength to be more effective, while it's the vice-versa if the density is lower. Honestly, I also tried simulated annealing when solving the lab and this last improvement wasn't so effective, but the others decreased the final cost in my case.
In case you change it as I said, you should modify a bit the code. In the if where you evaluate if the solution has to be taken, you cannot consider fitness[0] as an indicator of validity, so you should call the valid() function that makes the algorithm too expensive, or you can check if fitness[0] == UNIVERSE_SIZE (I would suggest this second strategy).  
```
I'm not detailing more over the comment I provided, it should be self-explainable (at least this is its aim).
The second student I had to review was Anil Bayram Gogebakan, who decided to use an Hill climber with a single mutation tweak. As I have previously done, I'll try to provide the most important part of the code and the comment I left in order to make everything understandable.  
This is the code:  
```python
solution = np.full(NUM_SETS, True)
history = [cost(solution)]
for n in tqdm(range(MAX_STEPS)):
    # TWEAK!
    new_solution = solution.copy()
    index = np.random.randint(0, NUM_SETS)
    new_solution[index] = not new_solution[index]

    # history.append(cost(new_solution))
    if cost(new_solution) < cost(solution) and valid(new_solution):
        solution = new_solution
    history.append(cost(new_solution))

# That's all...
ic(cost(solution))
ic(history.index(cost(solution)))

plt.figure(figsize=(14, 8))
plt.plot(
    range(len(history)),
    list(accumulate(history, min)),
    color="red",
)
_ = plt.scatter(range(len(history)), history, marker=".")
```
This is the comment I provided to my collegue:  
```
I try to comment and get you some feedbacks about the algorithm you delivered for the set-cover problem and to suggest some possible improvements on the results mantaining the same strategy you have used, that is hill climbing (with single mutation tweak currently).
First thing that I would suggest is to introduce a fitness and a tweak function, this is just for code readibility, it makes things easier for you as well while you are programming and testing possible scenarios.
Regarding the fitness, you are currently evaluating the goodness of a solution through the following line of code:
if cost(new_solution) < cost(solution) and valid(new_solution)
What I would do is to introduce a fitness function that returns a tuple (coverage,-cost), in a way that you are able to compare different solution easily. The real improvement you get by following this strategy consists in using the coverage instead of the validity. Indeed, by using the validity as a parameter, you are considering invalid solutions to be all the same, while actually they are not. The coverage represents the number of elements of the universe that are covered by your solution. At the end, you should get to a coverage == UNIVERSE_SIZE. If you don't want to use tuples and to introduce a fitness function you can check costs as you're doing right now and then, in an or condition, check if the coverage is higher. In reality, this would be different from the strategy I proposed, because you are not prioritizing coverage. So, definitely, I suggest to use tuples (they are checked following the positional order in python).
Another improvement I can suggest is to change the tweak function. Currently, it is perfroming a single tweak (one single element in the genotype is changed from true to false or vice-versa). When doing first experiments on the lab code, in my case, it came out that a multiple tweak stategy was much more effective! Start by changing an higher percentage of the sets by using a mask that changes a fixed percentage of elements of the genotype, and then you could also introduce a strength parameter to regulate the percentage of elements you change based on the density of the instance or based on the last results you obtained (Hill climbing with self-adaptive temperature/strength).
Last but not least, I suggest to compute the number of iterations based on the instance you are facing (i.e.: the parameters of the problem). If your algorithm struggles to compute the result for higher instances, probably you have to dynamically adapt the parameters of the algorithm itself based on the instance. In theory, as the problem space grows, the same should happen with the number of iterations, which means that you should start with a lower number of iterations and then increase them, but always making sure that you are able to execute the algorithm in a reasonable amount of time (find the trade-off you prefer between number of calls to the cost function and the result obtained).
I hope this will be somehow useful for you to improve the current result.
```
The last comment on the number of iterations comes from a note left by the author himself in the README, where he stated that he was not able to computer the result for the last instances that are computationally more heavy.  

## Received review Lab 1
Similarly to what happened in the first section, I will examin the reviews that has been opened through issues regardin the solution I proposed.
One of them was actually really helpful to improve the performance of my algorithm!
Let's start with the first one:
```
The algorithm itself is highly optimized, and I appreciate how you've combined various optimization techniques to improve performance, making the concept of hill climbing and its features easier to grasp.

One of the standout aspects of this solution is its adaptability. The way the iteration count scales with problem size, and how mutation strength adjusts based on problem density and recent progress tracking, shows a thoughtful approach. These elements are not only valuable for optimization but also for handling the different cases within the lab. Additionally, the balance between exploration and exploitation—starting with exploration and then refining through exploitation—adds to the effectiveness of the method. Offering two initialization strategies (empty set and greedy algorithm) is a great feature, as it demonstrates how different starting points affect the results. The empty set might lead to less polished solutions, but it helps avoid getting stuck in local optima.

Maybe adding some graphical content helps in following the process better
```
I will go briethly through this review because it didn't help me to improve the algorithm. Actually, graphical contents weren't so easy to be used in my case. Indeed, I start from a non-valid solution that takes time to converge, which means that you start getting something useful only when the entire universe is covered, and values are intentionally studied to make it happen towards the end of the execution. So, plotting the cost when the universe is not covered yet, is not meaningful. Anyway, I tried to use it to understand how many elements were added to the solution at each step and it was quite helpful. Anyway, thanks for the comment, I really appreciated what you said.  
Let's now move to the second review:
```
The solution is well organised the comments and clarifications are really helpful when reading the code especially the detailed read me file . The algorithm is well structured and I must compliment you for incorporating several optimization features together , it helped me to understand better hill climbing and all its features. I think what I liked most about the solution is its adaptivity .The iteration number depending on the problem size the adapting mutation strength with the problem density and recent progress that is tracked are useful not only in optimization but also dealing accordingly with all the instances of the lab combined this with the exploration exploitation balance where we first explore than exploit and refine makes for a well thought solution. You gave 2 versions of initialization (empty set or greedy algorithm ) I think this is a good way to see how starting with a greedy solution can refine our best result and empty set initialization may give less refined solutions but in trade of lower chances to get stuck in local optima .

Suggested enhancements: I would suggest a better visualization like a graph so we are able to follow the progress especially in fitness improvement during exploration and exploitation phases and with the adapting mutation strength. It would be helpful to see clearly how the algorithm is performing and where can it can be improved.

This could also help with the termination criteria. Currently the code has a fixed number of iterations per instance which can waste computation if the solution converges very fast. Adding a threshhold of improvement can be helpful.

One other enhancement can be in the separation of exploration and exploitation which is fixed into 20 % to 80%. If the solution improves fast in the beginning we can have less exploration or the other way around. Although making a lot of things self adaptive can be tricky and easier suggested than done :).

In general I think its a very good solution and I must compliment you.

```
This review, as I said before, was really helpful. I didn't get, when first developing the algorithm, that, at the beginning, exploitation was completely useless because we were too far from a local/global optima. I first share the modified version of the code and then comment it:
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
        if(np.sum(buffer)>=1):
            strength *= (1+DENSITY)
        else:
            break
    solution_perturbated = multiple_mutation_strength(best,strength*4) #a huge mutation->exploration
    for steps_local in range(number_iteration_exploitation):
        count_iteration+=1
        new_solution = multiple_mutation_strength(solution_perturbated,strength) 
        fitness_new = fitness(new_solution)
        buffer.append(fitness_new > solution_fitness)
        #We can try to reduce the number of iterations by stopping the inner cicle if we are not optaining good results
        if(steps_local % int(0.2*number_iteration_exploitation) == 0): #every 20% of the iterations we check if we are improving
            buffer_auxiliary = buffer[int(number_iteration_exploitation*0.2):]
            if(np.sum(buffer)==0):
                break
        if fitness_new > solution_fitness:
            solution_fitness = fitness_new
            solution_perturbated = new_solution
            best = new_solution.copy()
ic(count_iteration)
ic(fitness(best))
ic(valid(best))
ic(cost.calls)
```
I have introduced two variations with respect to the first delivered version. First of all, in the inner cycle, I check if there has been enancement in the last 20% of the internal iterations. If not, I break the cycle and try to perform exploration. In this way, as suggested by the reviewer, I am able to keep exploration and exploitation not fixed but variable depending on the context. Moreover, also the outer cycle had a number of fixed iterations. In the new version, I slightly changed the condition in the elif: If no improvements have been reached in a certain number of iterations (depending on the instance size), then, break. So, again, thanks to Xhoana Shkajoti I have been able to improve the results, in particular in terms of number of calls to the cost function (costs, actually, were already pretty low). I report here the updated results obtained with the new version of the algorithm:
| Instance | Cost | Number of steps |
| --------- | --------- | --------- |
| UNIVERSE_SIZE = 100, SET_SIZE = 10, DENSITY = 0.2            | 258.1275312074812   | 335   |
| UNIVERSE_SIZE = 1_000, SET_SIZE = 100, DENSITY =  0.2        | 8943.002719966002  | 282    |
| UNIVERSE_SIZE = 10_000 , SET_SIZE = 1_000 , DENSITY =  0.2   | 153947.43571873175  | 669    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.1  | 2412438.5118881054   | 2118    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.2  | 2852930.4020191683  | 186    |
| UNIVERSE_SIZE = 100_000 , SET_SIZE = 10_000 , DENSITY = 0.3  | 3029116.512182544 | 188    |

As we can observe, in the last two instances, costs are slightly more, but we are able to obtain them in less than 1/10 calls to the count function.