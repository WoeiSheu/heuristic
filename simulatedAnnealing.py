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

def create_new(solution):
    '''
    Create a new solution
    Exchange two elements in the original solution
    '''
    length = len(solution)
    i,j = np.random.choice(length,2)
    idx = solution[i]
    solution[i] = solution[j]
    solution[j] = idx


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

    # Use a list to represent the solution, the element of the list is the index of the city.
    solution = list(range(len(cities)))
    #solution = init_solution()
    #print(solution)

    # Record the fitness throughout the generation
    fitness_list = []

    # Generation
    generation = 0
    # Degree
    degree = 5000
    while generation < 3000 and degree > 1e-8:
        # For same degree, try multiple times
        # In order to avoid covergence too fast
        for _ in range(10):
            curr_fitness = calc_fitness(solution)
            prev_fitness = curr_fitness
            origin_solution = solution[:]
            create_new(solution)
            curr_fitness = calc_fitness(solution)
            
            delta_fitness = curr_fitness - prev_fitness
            # Metropolis
            if delta_fitness >= 0:
                r = np.random.rand()
                if np.exp(-delta_fitness/degree) < r:
                    solution = origin_solution[:]
                    fitness_list.append(prev_fitness)
                    continue
            fitness_list.append(curr_fitness)

        degree *= 0.98
        generation += 1

    #plt.title('Simulated Annealing')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.plot(fitness_list)
    print(np.min(fitness_list))

    return np.min(fitness_list)


if __name__ == "__main__":
    distances = []
    for _ in range(30):
        distance = run()
        distances.append(distance)
    print(distances)
    std = np.std(distances)
    print('Standard deviation is: {}'.format(std))