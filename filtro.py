# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib as npmatlib
import math
import matplotlib.pyplot as plt
from decimal import Decimal
from bisect import bisect_left

# Filter constants
e12_values = [1., 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
    
e24_values = [1., 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2., 2.2, 2.4, 2.7, 3., \
              3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, \
              9.1]
              
e96_values = [1., 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,\
              1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, \
              1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, \
              1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, \
              2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, \
              3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, \
              3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, \
              4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, \
              5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, \
              7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, \
              8.87, 9.09, 9.31, 9.53, 9.76]

res_exps = [3,4,5]
cap_exps = [-7,-8,-9]

res_vals = [x*(10**y) for x in e24_values for y in res_exps]
cap_vals = [x*(10**y) for x in e12_values for y in cap_exps]
res_vals.sort()
cap_vals.sort()

# Problem definition
num_dimensions = 4 # R2, R3, C4, C5
cap_min = 1e-9
cap_max = 8.2e-7
res_min = 1e3
res_max = 9.1e5

# Penalizing factors for gain, omega and Q
rlambda = 100000.0
rlambda_w = 100.0
rlambda_q = 100.0

# Target (desired) values for gain, omega and Q
gobj = 3.0
wp = 1000*2*math.pi
Qp = 1/math.sqrt(2.0)
invQp=1/Qp

err = 0.025
gmax=(1+err)*gobj
gmin=(1-err)*gobj
wmax=(1+err)*wp
wmin=(1-err)*wp
Qmax=(1+err)*Qp
Qmin=(1-err)*Qp

# AUXILIAR FUNCTIONS
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
       

def wheel_selection(P):
    r = np.random.uniform()
    C = np.cumsum(P)
    for i in range(0, len(C)):
        if C[i] > r:
            break
        
    j = max(0,i-1)
    return j

def find_discrete_neighbours(cont_val, isRes):
    vals = res_vals if isRes else cap_vals
    pos = bisect_left(vals, cont_val)
    if pos == 0:
        return [vals[0]]
    if pos == len(vals):
        return [vals[-1]]
    before = vals[pos - 1]
    after = vals[pos]
    return [before, after]

def find_discrete_neighbours2(cont_val, isRes):
    vals = res_vals if isRes else cap_vals
    pos = bisect_left(vals, cont_val)
    if pos == 0:
        return [vals[0], vals[1]]
    if pos == len(vals):
        return [vals[-1], vals[-2]]
    
    before = [vals[pos - 1]]
    after = [vals[pos]]
    if (pos - 2 > 0):
        before.append(vals[pos-2])
    if (pos + 1 < len(vals)):
        after.append(vals[pos+1])
    return before + after

def get_sol_info(r1, r2, r3, c4, c5):
    a = r1/r2
    b = r1/r3
    g = 1/a
    sens = (2 + abs(1-a+b) + abs(1+a-b))/(2*(1+a+b))
    omega = math.sqrt(a*b/(c4*c5))/r1
    invq = math.sqrt(c5/(c4*a*b))*(1+a+b)
    
    return (sens, g, 1./invq, omega)

# END AUXILIAR FUNCTIONS


# ACOR params
archive_size = 10
sample_size = 40
max_iterations = 200
int_factor = 0.5 # Intensification factor
zeta = 1.0 # Deviation-Distance ratio

def mainLoop(R1):
    # ---------------- COST FUNCTION -------------------
    def cost(arr):
        r1=R1
        r2, r3, c4, c5 = arr[0], arr[1], arr[2], arr[3]
        r2_range_OK = r2 > res_min and r2 < res_max
        r3_range_OK = r3 > res_min and r3 < res_max
        c4_range_OK = c4 > cap_min and c4 < cap_max
        c5_range_OK = c5 > cap_min and c5 < cap_max
        if r2_range_OK and r3_range_OK and c4_range_OK  and c5_range_OK:
            a = r1/r2
            b = r1/r3
            ganancia = 1/a
            sensibilidad = (2 + abs(1-a+b) + abs(1+a-b))/(2*(1+a+b))
            omega = math.sqrt(a*b/(c4*c5))/r1
            invq = math.sqrt(c5/(c4*a*b))*(1+a+b)
            costo = 1000*(sensibilidad-0.75)**2 + rlambda*(ganancia - gobj)**2 + rlambda_w*(math.log(omega/wp))**2 + rlambda_q*(math.log(invq/invQp))**2
        else:
            # Heavily penalize configurations that are not in the specified range
            costo = 1.0e11
        return costo
        # ---------------- END COST FUNCTION -------------------
    
    # Initialization
    filename = 'results_' + str(int(R1)) + '.txt'
    f = open(filename, 'w+')
    # Create Archive Table with archive_size rows and (num_dimensions + 1) columns
    empty_ant = np.empty([num_dimensions + 1])
    archive = npmatlib.repmat(empty_ant, archive_size, 1)
    
    #Initialize archive, right now it contains garbage
    for i in range(0, archive_size):
        for j in range(0,  num_dimensions + 1):
            if (j < 2):
                # Resistor
                archive[i][j] = np.random.uniform(res_min, res_max)
            elif (j < num_dimensions):
                # Capacitor
                archive[i][j] = np.random.uniform(cap_min, cap_max) 
            else:
                # Cost
                archive[i][j] = cost(archive[i][0:num_dimensions])
                
    # Sort it according to cost
    archive = archive[archive[:,num_dimensions].argsort()]
    initial_cost = archive[0][0:num_dimensions]
    f.write('Initial best population: ' + str(initial_cost)  + '\n')
    
    
    
    best_sol = archive[0][0:num_dimensions]
    # Array to hold best cost solutions
    best_cost = np.zeros([max_iterations])
    
    # Weights array
    w = np.empty([archive_size])
    for l in range(0, archive_size):
        f_factor = 1/(math.sqrt(2*math.pi)*int_factor*archive_size)
        s_factor = math.exp(-0.5*(l/(int_factor*archive_size))**2)
        w[l] = f_factor * s_factor
    
    # Selection probabilities
    p = w / np.sum(w)
    
    # ACOR Main Loop
    for it in range(0, max_iterations):
        # Means
        s = np.zeros([archive_size, num_dimensions])
        for l in range(0, archive_size):
            s[l] = archive[l][0:num_dimensions]
        
        # Standard deviations
        sigma = np.zeros([archive_size, num_dimensions])
        for l in range(0, archive_size):
            D = 0
            for r in range(0, archive_size):
                D += abs(s[l]-s[r])
            sigma[l] = zeta * D / (archive_size - 1)
            
        # Create new population array
        new_population = np.matlib.repmat(empty_ant, sample_size, 1)
        # Initialize solution for each new ant
        for t in range(0, sample_size):
            new_population[t][0:num_dimensions] = np.zeros([num_dimensions])
            
            for i in range(0, num_dimensions):
                # Select Gaussian Kernel
                l = wheel_selection(p)
                # Generate Gaussian Random Variable
                new_population[t][i] = s[l][i] + sigma[l][i]*np.random.randn()
                
            # Evaluation
            new_population[t][num_dimensions] = cost(new_population[t][0:num_dimensions])
            
        # Merge old population (archive) with new one
        merged_pop = np.concatenate([archive, new_population])
        # And sort it again
        merged_pop = merged_pop[merged_pop[:,num_dimensions].argsort()]
        # Store the bests in the archive and update best sol
        archive = merged_pop[:archive_size]
        best_sol = archive[0][0:num_dimensions]
        best_cost[it] = archive[0][num_dimensions]
        
        # Show iteration info
        f.write('Iteration %i, best cost found %s\n' % (it, format_e(Decimal(best_cost[it]))))
        f.write('  best solution: ' + str(best_sol)  + '\n')
        f.write('  cost best solution: ' + str(best_cost[it])  + '\n')
    
    initial_cost = archive[0][0:num_dimensions]
    f.write('Final best population: ' + str(initial_cost)  + '\n')
    
    neighbours = []
    neighbours.append(find_discrete_neighbours2(r1, True))
    k = 0
    for filter_comp in archive[0][0:num_dimensions]:
        isRes = k < 2
        this_neighbours = find_discrete_neighbours2(filter_comp, isRes)
        k += 1
        neighbours.append(this_neighbours)

    f.close()
    
#    plt.figure(figsize=(10,10))
#    plt.yscale('log')
#    plt.plot(range(1, max_iterations+1), best_cost)
#    plt.savefig('foo.png')
#    plt.show()
    
    return neighbours

f = open('final_solutions.txt', 'w+')
r1_vals = res_vals
all_sols = []
for r1 in r1_vals:
    print 'Running loop for R1 = ', r1
    discrete_neighbours = mainLoop(r1)
#    print discrete_neighbours
#    raw_input('Continue')
    for _r1 in discrete_neighbours[0]:
        for r2 in discrete_neighbours[1]:
            for r3 in discrete_neighbours[2]:
                for c4 in discrete_neighbours[3]:
                    for c5 in discrete_neighbours[4]:
                        tempt_sol = [_r1, r2, r3, c4, c5]
                        (sens, g, q, w) = get_sol_info(_r1, r2, r3, c4, c5)
    #                    print 'Sol: ', r1,r2,r3,c4,c5
    #                    print 'Sens: ', sens
    #                    print gmin, g, gmax
    #                    print Qmin, q, Qmax
    #                    print wmin, w, wmax
    #                    print '\n\n'
    #                    raw_input('Continue')
                        gOk = g > gmin and g < gmax
                        qOk = q > Qmin and q < Qmax
                        wOk = w > wmin and w < wmax
                        if (sens < 1 and gOk and qOk and wOk and not tempt_sol in all_sols):
                            all_sols.append(tempt_sol)
                            sol_str = [format_e(Decimal(x)) for x in tempt_sol]
                            f.write(str(tempt_sol)+'\t'+str(sens)+'\n')
f.close()