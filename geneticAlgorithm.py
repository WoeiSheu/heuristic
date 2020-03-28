import numpy as np
import matplotlib.pyplot as plt


class City():
    '''
    City model
    x: location in x axis
    y: location in y axis
    idx: idx of the city
    '''
    def __init__(self,x,y,idx):
        self.x   = float(x)
        self.y   = float(y)
        self.idx = int(idx)
    
    def distance(self,city):
        xd = self.x - city.x
        yd = self.y - city.y
        return np.ceil(np.sqrt( (xd**2 + yd**2)/10 ))

    def __repr__(self):
        return "The position of No. {} city is [{},{}]".format(self.idx,self.x,self.y)


cities = []

def load(filename):
    '''
    Load dataset of tsp problem.
    '''
    positions = []
    with open(filename,'r') as f:
        for line in f:
            positions.append(line.strip().split(' '))
    positions = positions[6:][:-1]

    global cities
    cities = []
    for position in positions:
        idx,x,y = position
        city = City(x,y,idx)
        cities.append(city)
    

def calc_fitness(solution):
    '''
    Calculate the fitness of the solution
    '''
    distance = 0
    for i in range(len(solution)-1):
        curr_city = cities[solution[i]]
        next_city = cities[solution[i+1]]
        distance += curr_city.distance(next_city)
    last_city = cities[solution[-1]]
    first_city = cities[solution[0]]
    distance += last_city.distance(first_city)
    
    return distance

def init_solution():
    start_city = np.random.randint(len(cities))
    visited_cities = [start_city]
    while len(visited_cities) != len(cities):
        min_distance = 1e10
        min_idx = -1
        for i in range(len(cities)):
            if i in visited_cities:
                continue
            distance = cities[visited_cities[-1]].distance(cities[i])
            if distance < min_distance:
                min_distance = distance
                min_idx = i
        visited_cities.append(min_idx)

    return np.array(visited_cities)


def run():
    load("dataset/att48.tsp")
    #print(cities)

    population_size = 40
    chromosome_len  = len(cities)
    crossover_rate  = 0.92
    mutation_rate   = 0.01
    max_generation  = 3000
    decrement_list  = np.arange(population_size,0,-1)
    cumulative_probabilities  = np.cumsum( decrement_list / decrement_list.sum() )

    # Init the population
    population = np.zeros((chromosome_len,population_size),dtype=int)
    parents    = np.zeros((chromosome_len,population_size),dtype=int)
    offsprings = np.zeros((chromosome_len,population_size),dtype=int)
    for i in range(population_size):
        population[:,i] = np.random.permutation(chromosome_len)
        #population[:,i] = init_solution()

    # Record the best fitness of the population in each generation
    best_fitness = np.empty(max_generation)
    best_solution = np.zeros((chromosome_len,max_generation), dtype=int)

    for generation in range(max_generation):
        ### Selection
        fitness_list = []
        for chromosome in population.T:
            fitness = calc_fitness(chromosome)
            fitness_list.append(fitness)
        fitness_list = np.array(fitness_list)
        sorted_idx = fitness_list.argsort()
        selected_idx = []
        for _ in range(population_size):
            j = np.argmax(cumulative_probabilities > np.random.rand())
            j = sorted_idx[j]
            selected_idx.append(j)
        
        #selected_idx = np.unique(selected_idx)

        parents                  = population[:,selected_idx]
        fitness_list             = fitness_list[selected_idx]
        best_fitness[generation] = fitness_list.min()
        idx_of_min               = fitness_list.argmin()
        best_solution[:,generation] = parents[:,idx_of_min]

        ### Crossover
        # Cycle Crossover
        for idx in range(0,population_size,2):
            father,mother = np.random.choice(population_size,2,replace=False)
            if np.random.rand() > crossover_rate:
                offsprings[:,idx]   = parents[:,father]
                offsprings[:,idx+1] = parents[:,mother]
                continue

            cross_point  = np.random.randint(chromosome_len)
            cross_points = [cross_point]
            for k in range(chromosome_len):
                cross_value = parents[:,father][cross_point]
                cross_point = np.where(parents[:,mother] == cross_value)[0][0]
                if cross_point == cross_points[0]:
                    break
                cross_points.append(cross_point)
            
            mask1 = np.ones(chromosome_len,dtype=int)
            mask1[cross_points] = 0
            mask2 = 1 - mask1
            
            offspring1 = mask1 * parents[:,father] + mask2 * parents[:,mother]
            offspring2 = mask2 * parents[:,father] + mask1 * parents[:,mother]
            offsprings[:,idx]   = offspring1
            offsprings[:,idx+1] = offspring2

        ### Mutation
        num_of_gene = population_size * chromosome_len
        num_of_mutation_gene = int(num_of_gene * mutation_rate)
        for _ in range(num_of_mutation_gene):
            mutation_idx = int(num_of_gene * np.random.rand())
            row          = int(mutation_idx / population_size)
            col          = int(mutation_idx - row*population_size)
            original_value = offsprings[row,col]
            mutation_value = np.random.randint(chromosome_len)
            exchange_point = np.where(offsprings[:,col] == mutation_value)[0][0]
            offsprings[row,col] = mutation_value
            offsprings[exchange_point,col] = original_value

        population = offsprings
    
    #plt.title('Genetic Algorithm')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.plot(best_fitness)
    #print(best_fitness)
    #print(best_fitness.min())
    #print(best_solution[:,best_fitness.argmin()])
    
    return best_fitness.min()


if __name__ == "__main__":
    distances = []
    for _ in range(30):
        distance = run()
        distances.append(distance)
    print(distances)
    std = np.std(distances)
    print('Standard deviation is: {}'.format(std))