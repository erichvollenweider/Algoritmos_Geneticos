import random
import math
from deap import base, creator, tools, algorithms


POP_SIZE = 20        # Tamaño de la población
GENS = 30            # Número de generaciones
CXPB = 0.8           # Probabilidad de cruce
MUTPB = 0.05         # Probabilidad de mutación
GENOME_LENGTH = 6    # Longitud de cada individuo (número de bits)

TARGET = (120, 200, 80)   # Color objetivo (R,G,B)


# ---------------------------
# Función de fitness
# ---------------------------

def fitness(individual):
    r, g, b = individual
    tr, tg, tb = TARGET
    dist = math.sqrt((r-tr)**2 + (g-tg)**2 + (b-tb)**2)
    return (-dist,)


# ---------------------------
# Crear población
# ---------------------------

def create_individual():
    return [random.randint(0, 255) for _ in range(3)]


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
            individual[i] = random.randint(0, 255)
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


#-------------------------- DEAP -------------------------

# Crear fitness y clase Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2. Toolbox: atributos, individuos, población
toolbox = base.Toolbox()

# Atributo: un entero entre 0 y 255
toolbox.register("attr_int", random.randint, 0, 255)

# Individuo: vector de 3 genes [R,G,B]
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, 3)

# Población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------
# Función de evaluación
# -----------------------------
def eval_color(individual):
    r, g, b = individual
    tr, tg, tb = TARGET
    dist = math.sqrt((r-tr)**2 + (g-tg)**2 + (b-tb)**2)
    return (-dist,)

toolbox.register("evaluate", eval_color)

# -----------------------------
# Operadores genéticos
# -----------------------------
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=255, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------
# Algoritmo principal
# -----------------------------
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.5, mutpb=0.2,
                        ngen=30, stats=stats,
                        halloffame=hof, verbose=True)

    best = hof[0]
    print(f"Mejor individuo encontrado: {best} → color {tuple(best)} "
          f"fitness {best.fitness.values[0]}")
    return best

if __name__ == "__main__":
    main()
