import random
import networkx as nx   # para comprobar conectividad
from deap import base, creator, tools, algorithms

# ------------------------------------------------
# Datos del problema
# ------------------------------------------------
NODOS = 5
# Lista de aristas posibles (sin dirección)
ARISTAS = [(0,1), (0,2), (0,3), (0,4),
           (1,2), (1,3), (1,4),
           (2,3), (2,4),
           (3,4)]

# Capacidad y costo para cada arista (mismo orden que ARISTAS)
MBPS  = [150, 275, 100, 350, 200, 115, 500, 425, 205, 300]
COSTO = [200, 500, 100, 550, 600, 150, 900, 500, 200, 200]

K_MAX = 6   # máximo número de conexiones permitidas

# ------------------------------------------------
# Configuración DEAP
# ------------------------------------------------
# Fitness multiobjetivo: max capacidad, min costo
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Gen: bit 0 o 1 (si la arista está incluida o no)
toolbox.register("attr_bool", random.randint, 0, 1)

# Un individuo: vector binario del tamaño de las aristas
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, len(ARISTAS))

# Una población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ------------------------------------------------
# Función de evaluación
# ------------------------------------------------
def evaluar(individuo):
    capacidad = 0
    costo = 0
    edges = []

    for i, bit in enumerate(individuo):
        if bit == 1:
            capacidad += MBPS[i]
            costo += COSTO[i]
            edges.append(ARISTAS[i])

    # penalización si excede K_MAX
    if sum(individuo) > K_MAX:
        return -1000, 10000  # muy malo

    # comprobar conectividad
    G = nx.Graph()
    G.add_nodes_from(range(NODOS))
    G.add_edges_from(edges)

    if not nx.is_connected(G):
        return -1000, 10000  # red no conectada

    return capacidad, costo

toolbox.register("evaluate", evaluar)

# ------------------------------------------------
# Operadores genéticos
# ------------------------------------------------
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# ------------------------------------------------
# Algoritmo principal
# ------------------------------------------------
def main():
    random.seed(42)

    pop = toolbox.population(n=50)
    NGEN = 40
    MU = 100
    LAMBDA = 40
    CXPB = 0.7
    MUTPB = 0.2

    # Evaluar población inicial
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Evolución con NSGA-II
    for gen in range(NGEN):
        offspring = tools.selNSGA2(pop, LAMBDA)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Aplicar cruzamiento y mutación
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

        for mut in offspring:
            if random.random() <= MUTPB:
                toolbox.mutate(mut)
                del mut.fitness.values

        # Evaluar los nuevos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Reemplazo NSGA-II
        pop = toolbox.select(pop + offspring, MU)

        # Mostrar mejor de la gen
        best = tools.selBest(pop, 1)[0]
        print(f"Gen {gen}: Mejor = {best}, Fitness = {best.fitness.values}")

    # Frente de Pareto final
    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    print("\nFrente de Pareto:")
    for ind in pareto:
        print(ind, " -> ", ind.fitness.values)

    return pop, pareto

if __name__ == "__main__":
    pop, pareto = main()
