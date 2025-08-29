import random
import math


POP_SIZE = 500        # Tamaño de la población
GENS = 100            # Número de generaciones
CXPB = 0.8           # Probabilidad de cruce
MUTPB = 0.05         # Probabilidad de mutación
GENOME_LENGTH = 8    # Longitud de cada individuo (número de bits)


def recorrer_por_columna(individual, i, j):

    k: int = 0
    while k < GENOME_LENGTH:
        if k == j:
            k += 1
        else:
            if individual[i][k] == 1:
                return False
            k += 1
    return True

def recorrer_por_fila(individual, i, j):

    k: int = 0
    while k < GENOME_LENGTH:
        if k == i:
            k += 1
        else:
            if individual[k][j] == 1:
                return False
            k += 1
    return True

def recorrer_por_diagonal(individual, i, j):
    # ↗ arriba-derecha
    k = 1
    N = GENOME_LENGTH
    while (i - k) >= 0 and (j + k) < N:
        if individual[i - k][j + k] == 1:
            return False
        k += 1

    # ↙ abajo-izquierda
    k = 1
    while (i + k) < N and (j - k) >= 0:
        if individual[i + k][j - k] == 1:
            return False
        k += 1

    # ↖ arriba-izquierda
    k = 1
    while (i - k) >= 0 and (j - k) >= 0:
        if individual[i - k][j - k] == 1:
            return False
        k += 1

    # ↘ abajo-derecha
    k = 1
    while (i + k) < N and (j + k) < N:
        if individual[i + k][j + k] == 1:
            return False
        k += 1

    return True

# ---------------------------
# Función de fitness
# ---------------------------

def fitness(individual):

    dist: int = 0
    count: int = 0

    # Contar cantidad de reinas
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            if individual[i][j] == 1:
                count += 1
                
    
    if count == GENOME_LENGTH:
        dist += 1000
    
    if (count - GENOME_LENGTH) < 0:
            dist += -(count * 100)

    if (count - GENOME_LENGTH) > 0:
        dist = dist + (GENOME_LENGTH - count) * 100

    for i in range(len(individual)):
        for j in range(len(individual[i])): 
            if individual[i][j] == 1: 
                if (
                    recorrer_por_columna(individual, i, j)
                    and recorrer_por_fila(individual, i, j)
                    and recorrer_por_diagonal(individual, i, j)
                ):
                    dist += 100
                else:
                    dist -= 500
    
    return dist


# ---------------------------
# Crear población
# ---------------------------

def create_individual():
    return [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(GENOME_LENGTH)]


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
    child1, child2 = [], []
    for row1, row2 in zip(p1, p2):
        if random.random() < CXPB:
            point = random.randint(1, GENOME_LENGTH - 1)
            c1 = row1[:point] + row2[point:]
            c2 = row2[:point] + row1[point:]
        else:
            c1, c2 = row1[:], row2[:]
        child1.append(c1)
        child2.append(c2)
    return child1, child2




def mutate(individual):
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            if random.random() < MUTPB:
                individual[i][j] = 1 - individual[i][j]
    return individual


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
def print_matrix(individual):
    for row in individual:
        print(" ".join(str(x) for x in row))

print("Mejor individuo encontrado (matriz):")
print_matrix(best)
print("Fitness =", fitness(best))
