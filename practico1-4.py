import random
import math


POP_SIZE = 20        # Tamaño de la población
GENS = 30            # Número de generaciones
CXPB = 0.8           # Probabilidad de cruce
MUTPB = 0.05         # Probabilidad de mutación
GENOME_LENGTH = 30   # Longitud de cada individuo (número de bits)



# ---------------------------
# Función de fitness
# ---------------------------

def decode_individual(individual):
    return [(individual[i], individual[i+1]) for i in range(0, len(individual), 2)]

def fitness(individual):
    coords = decode_individual(individual)
    dist = 0
    n = len(coords)
    for i in range(n - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        dist += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # Cerrar el ciclo
    x1, y1 = coords[-1]
    x2, y2 = coords[0]
    dist += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return (-dist,)


# ---------------------------
# Crear población
# ---------------------------

def create_individual():
    return [random.randint(0, 14) for _ in range(GENOME_LENGTH)]


def create_population():
    return [create_individual() for _ in range(POP_SIZE)]


# ---------------------------
# Operadores genéticos
# ---------------------------

def selection(population):
    # Selección por torneo
    k = 3
    selected = []
    for _ in range(POP_SIZE):
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=fitness)
        selected.append(winner)
    return selected

def crossover(p1, p2):
    # Combinación de un punto
    if random.random() < CXPB:
        point = random.randint(1, GENOME_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTPB:
            individual[i] = random.randint(0, 14)
    return individual



individual = create_individual()
print("Individual versión clase: ", individual)
print("MaxOnes versión clase: ", fitness(individual))


# ---------------------------
# Algoritmo principal
# ---------------------------

def genetic_algorithm():

    population = create_population()
    for gen in range(GENS):
        # Evaluar y mostrar el mejor
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Mejor = {population[0]} Fitness = {fitness(population[0])}")
        # Selección
        selected = selection(population)
        # Reproducción
        next_gen = []

        for i in range(0, POP_SIZE, 2):
            offspring1, offspring2 = crossover(selected[i], selected[i+1])
            next_gen.append(mutate(offspring1))
            next_gen.append(mutate(offspring2))

        population = next_gen

    return max(population, key=fitness)

# ---------------------------
# Ejecutar
# ---------------------------

best = genetic_algorithm()
print(f"Mejor individuo encontrado: {best}, Fitness = {fitness(best)}")
