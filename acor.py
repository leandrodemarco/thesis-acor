# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib as npmatlib
import math
import matplotlib.pyplot as plt
from decimal import Decimal

# Problem definition
num_dimensions = 10 # 3 resistors and 2 capacitors
cap_min = -10 #1e-9
cap_max = 10 #8.2e-7
res_min = -10 #1e3
res_max = 10 #9.1e5
var_min = 0.5
var_max = 1.5


# ---------------- COST FUNCTIONS -------------------
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def cost_sphere(arr):
    return sum([x**2 for x in arr])
    
def cost_plane(arr):
    return arr[0] if all(v > var_min and v < var_max for v in arr) else 10

def cost_diagplane(arr):
    return np.average(arr)

def cost_ellipsoid(arr):
    n = len(arr)
    s = 0.0
    i = 1
    for x in arr:
        s += x *(100 ** ((i-1)/(n-1)))
        i += 1
        
    return s

def cost_cigar(arr):
    return arr[0]**2 + 10**4 * (sum([x**2 for x in arr]))

def cost_tablet(arr):
    return (100*arr[0])**2 + sum([x**2 for x in arr])

def cost_rosenbrock(arr):
    s = 0.0
    for i in range(0, len(arr)-1):
        x_i = arr[i]
        x_i1 = arr[i+1]
        new_s = 100 * (x_i**2 - x_i1)**2 + (x_i - 1)**2
        s += new_s
        
    return s

# ---------------- END COST FUNCTIONS -------------------
        

def wheel_selection(P):
    r = np.random.uniform()
    C = np.cumsum(P)
    for i in range(0, len(C)):
        if C[i] > r:
            break
        
    j = max(0,i-1)
    return j

# ---------------------------------------------------

# Select const function
str_func = raw_input("""Choose cost function\n1-Sphere\n2-Plane\n3-Diagonal Plane\n4-Ellipsoid\n5-Cigar\n6-Tablet\n7-Rosenbrock: """)
ifunc = int(str_func)

func_name = 'Sphere'
cost = cost_sphere # Default
if (ifunc == 2):
    func_name = 'Plane'
    cost = cost_plane
elif (ifunc == 3):
    func_name = 'Diagonal Plane'
    cost = cost_diagplane
elif (ifunc == 4):
    func_name = 'Ellipsoid'
    cost = cost_ellipsoid
elif (ifunc == 5):
    func_name = 'Cigar'
    cost = cost_cigar
elif (ifunc == 6):
    func_name = 'Tablet'
    cost = cost_tablet
elif (ifunc == 7):
    func_name = 'Rosenbrock'
    cost = cost_rosenbrock


# ACOR params
archive_size = 10
sample_size = 40
max_iterations = 1000
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
    for j in range(0, num_dimensions + 1):
        if (j < 3):
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

plt.figure(figsize=(10,10))
plt.plot(range(1, max_iterations+1), best_cost)
#plt.show()
f.close()
#plt.axis([0, max_iterations+1, 0, max(best_cost)])

    