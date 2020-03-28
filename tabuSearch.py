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

def get_neighbors(solution):
    '''
    Get neighbors
    '''
    length = len(solution)
    neighbors = []
    for i in range(length):
        for j in range(i+1, length):
            neighbor = np.copy(solution)
            k = neighbor[i]
            neighbor[i] = neighbor[j]
            neighbor[j] = k
            neighbors.append(neighbor)
    
    neighbors = np.array(neighbors)
    nums = len(neighbors)
    #return neighbors[np.random.choice(nums,length*5,replace=False),:]
    return neighbors[np.random.choice(nums,30,replace=False),:]

def exists_in(tabu_list, candidate):
    for solution in tabu_list:
        if np.array_equal(solution,candidate):
            return True
    return False

def get_neighbors_with(tabu_list, solution):
    '''
    Get neighbors with tabu_list
    '''
    length = len(solution)
    neighbors = []
    for i in range(length):
        for j in range(i+1, length):
            neighbor = np.copy(solution)
            k = neighbor[i]
            neighbor[i] = neighbor[j]
            neighbor[j] = k
            if exists_in(tabu_list, neighbor):
                continue
            neighbors.append(neighbor)
    
    neighbors = np.array(neighbors)
    return neighbors
 
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
    #best_solution = np.array(range(len(cities)))
    best_solution = init_solution()
    #print(solution)
    best_candidate = best_solution

    best_fitness = calc_fitness(best_solution)

    #tabu_size  = int(len(cities)*20)
    tabu_size = 300
    tabu_list = []
    tabu_list.append(best_candidate)

    fitness_list = []

    # Generation
    for generation in range(3000):
        candidates = get_neighbors(best_candidate)
        #candidates = get_neighbors_with(tabu_list,best_candidate)
        best_candidate = candidates[0]
        best_candidate_fitness = calc_fitness(best_candidate)
        for candidate in candidates:
            if exists_in(tabu_list,candidate):
                continue
            fitness = calc_fitness(candidate)
            if fitness < best_candidate_fitness:
                best_candidate = candidate
                best_candidate_fitness = fitness
        tabu_list.append(best_candidate)
        if len(tabu_list) > tabu_size:
            tabu_list = tabu_list[1:]
        if best_candidate_fitness < best_fitness:
            best_solution = best_candidate
            best_fitness = best_candidate_fitness
        
        fitness_list.append(best_fitness)
        print(generation,best_fitness)
    
    #plt.title('Tabu Search')
    #plt.xlabel('Generation')
    #plt.ylabel('Fitness')
    #plt.plot(fitness_list)
    print(best_fitness)

    return best_fitness


if __name__ == "__main__":
    distances = []
    for _ in range(30):
        distance = run()
        distances.append(distance)
    print(distances)
    std = np.std(distances)
    print('Standard deviation is: {}'.format(std))