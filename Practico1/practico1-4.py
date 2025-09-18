import random
import math


POP_SIZE = 1000        # Tamaño de la población
GENS = 30            # Número de generaciones
CXPB = 0.8           # Probabilidad de cruce
MUTPB = 0.05         # Probabilidad de mutación
GENOME_LENGTH = 7   # Longitud de cada individuo (número de bits)

RECORRIDOS = [(0,1), (1,0), (0,2), (2,0), (0,5), (5,0), (1,3), (3,1), (2,3), (3,2), (2,4), (4,2), (3,5), (5,3), (4,5), (5,4)]
DISTANCIAS = [7, 7, 9, 9, 14, 14, 10, 10, 2, 2, 8, 8, 6, 6, 9, 9]

"""
Los individuos representan una secuencia de ciudades a visitar.    if (individual[i], individual[j]) not in RECORRIDOS:
"""

def filtrar_ciudades(individual):
    # Primer igual al último
    if individual[0] != individual[-1]:
        return False
    
    # El nodo inicial no debe repetirse en el medio
    if individual[0] in individual[1:-1]:
        return False
    
    # El resto de nodos (excluyendo inicio/fin) no deben repetirse
    camino = individual[1:-1]
    if len(camino) != len(set(camino)):
        return False
    
    return True


def es_valido(individual):
    n = len(individual)
    for i in range(n - 1):
        if (individual[i], individual[i+1]) not in RECORRIDOS:
            return False
    return True


# ---------------------------
# Función de fitness
# ---------------------------

# [0 1 3 2 4 5 0]
# [0 1 3 5 4 2 0]
# [0 2 4 5 3 1 0]
# [0 5 4 2 3 1 0]
# [1 0 2 4 5 3 1]
# [1 3 2 4 5 0 1]
# [1 3 5 4 2 0 1]
# [1 0 5 4 2 3 1]
# [2 0 1 3 5 4 2]
# [2 4 5 3 1 0 2]
# [2 4 5 0 1 3 2]
# [3 1 0 2 4 5 3]
# [3 1 0 5 4 2 3]
# [3 2 4 5 0 1 3]
# [3 5 4 2 0 1 3]
# [4 2 0 1 3 5 4]
# [4 2 3 1 0 5 4]
# [4 5 3 1 0 2 4]
# [4 5 0 1 3 2 4]
# [5 0 1 3 2 4 5]
# [5 3 1 0 2 4 5]
# [5 4 2 0 1 3 5]
# [5 4 2 3 1 0 5]


def fitness(individual):
    if not es_valido(individual):
        return -10000  # penalización fuerte

    dist = 0
    n = len(individual)
    for i in range(n - 1):
        par = (individual[i], individual[i+1])
        indice = RECORRIDOS.index(par)
        dist += DISTANCIAS[indice]
    return -dist


# ---------------------------
# Crear población
# ---------------------------

def create_individual():
    return [random.randint(0, 5) for _ in range(GENOME_LENGTH)]


def create_population():
    return [create_individual() for _ in range(POP_SIZE)]


# ---------------------------
# Operadores genéticos
# ---------------------------

def selection(population):
    k = min(3, len(population))  # nunca pide más de los que hay
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
    for i in range(1, len(individual)-1):
        if random.random() < MUTPB:
            individual[i] = random.randint(0, 5)
    return individual


individual = create_individual()
print("Individual versión clase: ", individual)
print("MaxOnes versión clase: ", fitness(individual))


# ---------------------------
# Algoritmo principal
# ---------------------------

def genetic_algorithm():

    population = create_population()
    population = [ind for ind in population if filtrar_ciudades(ind)]
    #population = [ind for ind in population if es_valido(ind)]
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

        next_gen = [ind for ind in next_gen if filtrar_ciudades(ind)]
        population = next_gen

    return max(population, key=fitness)

# ---------------------------
# Ejecutar
# ---------------------------

best = genetic_algorithm()
print(f"Mejor individuo encontrado: {best}, Fitness = {fitness(best)}")
