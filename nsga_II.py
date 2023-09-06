import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Link Google Collab: https://colab.research.google.com/drive/1dNnuJI8UeRRTZXkiyOLBWk52dmm-4DdP?usp=sharing

def objectives(x, y, problem_type="VIE1"):
    if problem_type == "VIE1":
        f1 = x**2 + (y - 1)**2
        f2 = x**2 + (y + 1)**2 + 1
        f3 = (x - 1)**2 + y**2 + 2
    elif problem_type == "VIE2":
        f1 = (((x - 2) ** 2) / 2) + (((y + 1) ** 2) / 13) + 3
        f2 = (((x + y - 3) ** 2) / 36) + (((-x + y + 2) ** 2) / 8) - 17
        f3 = (((x + 2 * y - 1) ** 2) / 175) + (((2 * y - x) ** 2) / 17) - 13
    elif problem_type == "VIE3":
        f1 = result1 = 0.5 * (np.power(x, 2) + np.power(y, 2)) + np.sin(np.power(x, 2) + np.power(y, 2))
        f2 = (((3 * x - 2 * y + 4) ** 2) / 8) + (((x - y + 1) ** 2) / 27) + 15
        f3 = (1 / (np.power(x, 2) + np.power(y, 2) + 1)) + 1.1 * np.exp(-1 * (np.power(x, 2) + np.power(y, 2)))
    else:
        raise ValueError("Unknown problem")
    return [f1, f2, f3]

def dominate(a, b):
    return all(a_i <= b_i for a_i, b_i in zip(a, b)) and any(a_i < b_i for a_i, b_i in zip(a, b))

def fast_non_dominated_sort(P):
    fronts = [[]]
    S = [[] for _ in P]
    n = [0] * len(P)
    
    for p, p_val in enumerate(P):
        for q, q_val in enumerate(P):
            if dominate(p_val, q_val):
                S[p].append(q)
            elif dominate(q_val, p_val):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]

def crowding_distance_assignment(I, P):
    l = len(I)
    distance = [0] * l
    distance[0], distance[-1] = float('inf'), float('inf')
    
    for m in range(len(P[0])):
        I.sort(key=lambda i: P[i][m])
        for i in range(1, l - 1):
            distance[i] += (P[I[i+1]][m] - P[I[i-1]][m])
    return distance


def select_by_tournament(P, rank, distance):
    pool_size = len(P)
    pool = np.random.choice(pool_size, pool_size, replace=True)
    selected = []
    
    for i in range(0, len(pool) - 1, 2):
        p1, p2 = pool[i], pool[i+1]
        if rank[p1] < rank[p2] or (rank[p1] == rank[p2] and distance[p1] > distance[p2]):
            selected.append(P[p1])
        else:
            selected.append(P[p2])
    
    return selected

def crossover_and_mutation(parents):
    children = []
    for parent1, parent2 in zip(parents[::2], parents[1::2]):
        child1 = [(p1 + p2) / 2 + np.random.normal(0, 0.1) for p1, p2 in zip(parent1, parent2)]
        child2 = [(p1 + p2) / 2 + np.random.normal(0, 0.1) for p1, p2 in zip(parent1, parent2)]
        children.extend([tuple(child1), tuple(child2)])
    return children

def NSGAII(problem_type="VIE1"):
    pop_size = 100
    generations = 400
    if problem_type == "VIE1":
        x_min, x_max = -2, 2
        y_min, y_max = -2, 2
    elif problem_type == "VIE2":
        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
    elif problem_type == "VIE3":
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
    
    P_x = np.random.uniform(x_min, x_max, pop_size)
    P_y = np.random.uniform(y_min, y_max, pop_size)
    P = list(zip(P_x, P_y))
    Q = None
    
    for gen in range(generations):
        R = P + Q if Q is not None else P
        F_R = [objectives(x, y, problem_type) for x, y in R]
        
        fronts = fast_non_dominated_sort(F_R)
        
        next_P = []
        next_Q = []
        rank = [0 for _ in range(len(R))]
        distance = [0 for _ in range(len(R))]
        
        for i, front in enumerate(fronts):
            curr_distance = crowding_distance_assignment(front, F_R)
            for idx in front:
                rank[idx] = i
                distance[idx] = curr_distance[front.index(idx)]
                
            if len(next_P) + len(front) <= pop_size:
                next_P.extend([R[i] for i in front])
            else:
                sorted_front = [front[i] for i in np.argsort(curr_distance)[::-1]]
                next_P.extend([R[i] for i in sorted_front[:pop_size - len(next_P)]])
                break
        
        selected = select_by_tournament(R, rank, distance)
        
        Q = crossover_and_mutation(selected)
        Q = [(np.clip(x, x_min, x_max), np.clip(y, y_min, y_max)) for x, y in Q]
        P = next_P
        
    F_P = [objectives(x, y, problem_type) for x, y in P]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([f[0] for f in F_P], [f[1] for f in F_P], [f[2] for f in F_P])
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    plt.title('Pareto Front')
    plt.show()

if __name__ == "__main__":
    NSGAII("VIE1")
    NSGAII("VIE2")
    NSGAII("VIE3")