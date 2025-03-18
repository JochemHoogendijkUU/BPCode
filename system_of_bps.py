#! /usr/bin/env python3
from random import Random
from collections import defaultdict, deque
import numpy as np
from math import exp
import matplotlib.pyplot as plt

# Function hierarchy in the code
# 1. Main function is ApproximateSolution, which calls GenerateTree several times
# 2. GenerateTree calls GenerateSubTree several times while growing the random tree
# 3. GenerateSubTree calls GenerateLevel1 GenerateLevel2 several times when building a subtree component
# 4. TreeFactor computes the associated value of a subtree
# 5. BinSearch tree is a binary search in the tree that looks up the value associated to a specific point in time.

def ApproximateSolution(rng, x, Ntrees, t_interval, type_dict):
    # Takes:
    # - x:          an m-dimensional vector taking values in [0, \infty)^m
    # - Ntrees:     the number of trees used to approximate the solution
    # - t_interval: numpy array denoting time
    # - type_dict:  defaultdict containing all types with their coefficients

    # Returns:
    # - a list of time and associated u(t, x) approximation

    # Procedure outline:
    #   - Convert type_dict to a modified coefficient dict associated to x
    #   - For each component generate Ntree trees, for each time point, average the trees so far
    #   - Return a sequence of vectors indexed by time

    t_cutoff = t_interval[-1]
    alpha_cutoff = 1000000
    m = len(x)

    #convert coefficients
    l1_types = GetLevelTypes(m, 1, type_dict)                   #Obtain level 1 types
    l2_types = GetLevelTypes(m, 2, type_dict)                   #Obtain level 2 types
    sol_type_dict = TypesAtX(x, l1_types, l2_types, type_dict)
    
    u = []
    for k in range(1, m + 1):
        u_k = []
        trees_k = [GenerateTree(rng, k, t_cutoff, alpha_cutoff, l1_types, l2_types, sol_type_dict) for n in range(Ntrees)]
        for t in t_interval:
            u_k_t = 0
            for tree_k in trees_k:
                index = BinSearchTree(t, tree_k)
                u_k_t += tree_k[index][1]
            u_k_t /= Ntrees
            u_k.append(u_k_t)
        u.append(u_k)
    return u

def GetLevelTypes(dim, level, type_dict):
    res = [[] for _ in range(dim)]
    for k, v in type_dict.items():
        if k[0] == level:
            res[k[1] - 1].append(k)
    return res

def TypesAtX(x, l1_types, l2_types, type_dict):
    transformed_type_dict = defaultdict(float)
    q = [1.0/len(l1k) for l1k in l1_types]
    phi = [1.0/len(l2k) for l2k in l2_types]
    for k, v in type_dict.items():
        level = k[0]
        if level == 1:
            dot = sum([pair[0] * pair[1] for pair in zip(k[2:], x)])
            new_coeff = v/(q[k[1] - 1]) * exp(-1.0 * dot)
        elif level == 2:
            new_coeff = (v + 1.0)/(phi[k[1] - 1])
        transformed_type_dict[k] = new_coeff
    return transformed_type_dict

def GenerateTree(rng, component, t_cutoff, alpha_cutoff, l1_types, l2_types, coef):
    #Specifications:
    #
    # Takes:
    # - rng:            python random number generator
    # - component:      The starting root component
    # - t_cutoff:       The maximal time value up to which we want to grow the tree
    # - alpha_cutoff:   the maximum level-1 mass we allow
    # - l1_types:       -----
    # - l2_types:       -----   
    # - coef:           A dictionary which tracks all the factors associated to each type. Note that x is included in this already

    # Returns:
    # - a list of tuples of size 2 which gives all the times at which a tree value changes

    # Procedure outline:
    #   select a root type
    #   While t < t_cutoff
    #       1. Find next point in time when a subtree is added
    #       2. Add subtree

    root_type = rng.choice(l1_types[component - 1])
    tree_value = coef[root_type]
    tree_alpha = list(root_type[2:])
    t = 0
    res = [(t, tree_value)]

    while(t < t_cutoff):
        if(sum(tree_alpha) > alpha_cutoff):
            break
        #Select point in time when subtree is added
        t += rng.expovariate(1.0/sum(tree_alpha))
        if(t > t_cutoff):
            break
        #Select level 2 type from which subtree is added
        added_types = defaultdict(int)
        l2_sub_root = Samplel2SubRoot(rng, tree_alpha, l2_types)
        added_types[l2_sub_root] += 1
        #Spawn subtree from level 2 type, increase tree alpha
        tree_alpha = GenerateSubTree(rng, tree_alpha, added_types, l2_sub_root, t, alpha_cutoff, l1_types, l2_types)
        #Compute the total tree factor from the subtree and multiply with current one
        tree_value *= TreeFactor(added_types, coef)
        res.append((t, tree_value))
    return res

def TreeFactor(type_count, coef):
    res = 1
    for k, v in type_count.items():
        res *= coef[k]**v
    return res

def Samplel2SubRoot(rng, tree_alpha, l2_types):
    m = len(tree_alpha)
    sum_alpha = sum(tree_alpha)
    prob_alpha = [alpha_i/sum_alpha for alpha_i in tree_alpha]
    pre_type = rng.choices(list(range(1, m + 1)), weights = prob_alpha)
    res = rng.choice(l2_types[pre_type[0] - 1])
    return res

def GenerateSubTree(rng, tree_alpha, new_types, root_type, t, alpha_cutoff, l1_types, l2_types):
    # Takes:
    # - rng:            the numpy random number generator
    # - tree_alpha:     a list which contains the sum of all alpha vectors in the tree 
    # - new_types:      a dictionary which tracks the number of types
    # - time:           the time parameter
    # - alpha_cutoff:   maximum level 1 mass in the tree
    # - l1_types:       a list of lists of tuples that contains all possible level 2 types
    #                   where list i contains all types with first index i
    # - l2_types:       a list of lists of tuples that contains all possible level 2 types
    #                   where list i contains all types with first index i

    #Modifies:
    # - new_trees to add the new_types

    # Returns:
    # - a modified tree_alpha
    m = len(tree_alpha)

    l1q = deque()
    l2q = deque()
    l2q.append(root_type)
    new_types[root_type] += 1
    while(l1q or l2q):
        if(sum(tree_alpha) > alpha_cutoff):
            break
        # Generate level 1 types
        while(l2q):
            top = l2q.popleft()
            new_l1_types = GenerateLevel1FromLevel2(rng, m, top, l1_types)
            for tp in new_l1_types:
                new_types[tp] += 1
                for i in range(1, m + 1):
                    tree_alpha[i - 1] += tp[1 + i]
            l1q.extend(new_l1_types)
        # Generate level 2 types
        while(l1q):
            top = l1q.popleft()
            new_l2_types = GenerateLevel2FromLevel1(rng, m, t, top, l2_types)
            for tp in new_l2_types:
                new_types[tp] += 1
            l2q.extend(new_l2_types)
    return tree_alpha


def GenerateLevel1FromLevel2(rng, dim, l2_type, l1_types):
    # Takes:
    # - rng:        the numpy random number generator
    # - l2_type:    a level 2 type encoded by a tuple
    # - l1_types:   a list of lists of tuples that contains all possible level 1 types,
    #               where list i contains all types with first index i.

    # Returns:
    # - a list of tuples, the tuples being the generated level 1 types
    level = l2_type[0]
    assert level == 2, "Wrong level, input level 1 type where there should have been level 2"    
    res = []
    for i in range(1, dim+1):
        beta_i = l2_type[i + 1]
        res += rng.choices(l1_types[i - 1], k = beta_i)    #Choose from l1_types, beta_i number of times 
    return res

def GenerateLevel2FromLevel1(rng, dim, t, l1_type, l2_types):
    # Takes:
    # - rng:        the numpy random number generator
    # - t:          the time parameter
    # - l1_type:    a level 1 type encoded by a tuple
    # - l2_types:   a list of lists of tuples that contains all possible level 2 types
    #               where list i contains all types with first index i.

    # Returns:
    # - a numpy array of tuples, the tuples being the generated l2_types
    level = l1_type[0]
    assert level == 1, "Wrong level, input level 1 type where there should have been level 2"    
    res = []
    for i in range(1, dim+1):
        #generate the pre-type
        alpha_i = l1_type[i + 1]
        n_pre_type_i = np.random.poisson(t * alpha_i)

        #generate the actual types
        res += rng.choices(l2_types[i - 1], k = n_pre_type_i)
                    
    return res

def BinSearchTree(time, tree):
    left = 0
    right = len(tree) - 1
    while(left <= right):
        mid = left + (right - left) // 2
        if tree[mid][0] == time:
            return mid
        
        if tree[mid][0] > time:
            right = mid - 1 
        elif tree[mid][0] < time:
            left = mid + 1
    #print(f"Time at right is {tree[right][0]}. Actual time is {time}") 
    return right

if __name__ == "__main__":
    rng = Random()
    rng.seed(1)

    N_trees = 5120000
    times = np.linspace(0, 0.2, 20)

    # Test case 1 with the following initial condition and nonlinearity:
    # - u_1(0, x) = 0.5 e^{-x_1} + 0.5 e^{-x_2}
    # - u_2(0, x) = -0.5 e^{-x_1} + 1.5 e^{- 2 x_2}
    # - F_1(s) = s_1 + s_2
    # - F_2(s) = s_1 * s_2
    # x = [1, 1]
    t1_td = defaultdict(float)
    # type vectors are encoded as tuples (level, component, mass_1, ..., mass_m)
    t1_td[(1, 1, 1, 0)] = 0.5
    t1_td[(1, 1, 0, 1)] = 0.5
    t1_td[(1, 2, 1, 0)] = -0.5
    t1_td[(1, 2, 0, 2)] = 1.5
    t1_td[(2, 1, 1, 0)] = 1
    t1_td[(2, 1, 0, 1)] = 1
    t1_td[(2, 2, 1, 1)] = 1
    # Need to add implementation for the above conditions
    test_1_x = (1, 1)
    test_1_u = ApproximateSolution(rng, test_1_x, N_trees, times, t1_td)

    #fig, ax = plt.subplots()

    #ax.plot(times, test_1_u[0])
    #ax.plot(times, test_1_u[1])

    #ax.set_xlabel('t')
    #ax.set_ylabel('u_1(., (1, 1))')
    #ax.set_title('Plot of u_1(. (1, 1))')
    #plt.show()
    # We can't really plot a 2 by 2 function, so we will have to be content with either coordinate
    
    # Test case 2 with the following initial condition and nonlinearity:
    # - u_1(0, x) = 0.5 e^{-x_1} + 0.5 e^{-x_2}
    # - u_2(0, x) = -0.5 e^{-x_1} + 1.5 e^{- 2 x_2}
    # - F_1(s) = 1
    # - F_2(s) = 1
    #
    # t2_td = defaultdict(float)

    # Test case 3 with the following initial condition and nonlinearity:
    # - u_1(0, x) = 0.5 e^{-x_1} + 0.5 e^{-x_2}
    # - u_2(0, x) = -0.5 e^{-x_1} + 1.5 e^{- 2 x_2}
    # - F_1(s) = 1
    # - F_2(s) = 0
