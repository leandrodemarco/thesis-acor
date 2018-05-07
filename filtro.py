# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib as npmatlib
import math
import matplotlib.pyplot as plt
from decimal import Decimal

# Problem definition
num_dimensions = 4 # R2, R3, C4, C5
cap_min = 1e-9
cap_max = 8.2e-7
res_min = 1e3
res_max = 9.1e5

R1=11000. #R1 is an input var

# Penalizing factors for gain, omega and Q
rlambda = 100000.0
rlambda_w = 100.0
rlambda_q = 100.0

# Target (desired) values for gain, omega and Q
gobj = 3.0
wp = 1000*2*math.pi
Qp = 1/math.sqrt(2.0)
invQp=1/Qp

#err = 0.025
#gmax=(1+err)*gobj
#gmin=(1-err)*gobj
#wmax=(1+err)*wp
#wmin=(1-err)*wp
#Qmax=(1+err)*Q
#Qmin=(1-err)*Q

# ---------------- COST FUNCTION -------------------
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
    
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
        

def wheel_selection(P):
    r = np.random.uniform()
    C = np.cumsum(P)
    for i in range(0, len(C)):
        if C[i] > r:
            break
        
    j = max(0,i-1)
    return j

# ---------------------------------------------------


func_name = 'filtro'


# ACOR params
archive_size = 10
sample_size = 40
max_iterations = 100
q = 0.5 # Intensification factor
zeta = 1.0 # Deviation-Distance ratio

# Initialization
f = open('results.txt', 'w+')
f.write('COST FUNCTION ' + func_name + '\n')
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
#print 'Initial archive:\n', archive
#raw_input('Enter')
initial_cost = archive[0][0:num_dimensions]
f.write('Initial best population: ' + str(initial_cost)  + '\n')



best_sol = archive[0][0:num_dimensions]
# Array to hold best cost solutions
best_cost = np.zeros([max_iterations])

# Weights array
w = np.empty([archive_size])
for l in range(0, archive_size):
    f_factor = 1/(math.sqrt(2*math.pi)*q*archive_size)
    s_factor = math.exp(-0.5*(l/(q*archive_size))**2)
    w[l] = f_factor * s_factor

# Selection probabilities
p = w / np.sum(w)
#print 'Selection probs: \n', p

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
#    print 'New population: \n', new_population
#    raw_input('Continue')
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
f.close()

plt.figure(figsize=(10,10))
plt.yscale('log')
#plt.axis([0, max_iterations+1, 0, max(best_cost)])
plt.plot(range(1, max_iterations+1), best_cost)
plt.savefig('foo.png')
plt.show()
