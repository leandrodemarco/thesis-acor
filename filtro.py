# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib as npmatlib
import math
from Utils import gobj,wp,Qp,invQp,wheel_selection,res_vals,cap_vals, \
                  get_sol_info, is_sol, is_soft_sol

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

err = 0.025
gmax=(1+err)*gobj
gmin=(1-err)*gobj
wmax=(1+err)*wp
wmin=(1-err)*wp
Qmax=(1+err)*Qp
Qmin=(1-err)*Qp


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
    
    #best_sol = archive[0][0:num_dimensions]
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
                aux = s[l][i] + sigma[l][i]*np.random.randn()
                if (i < 2):
                    # Resistor
                    new_population[t][i] = res_vals[(np.abs(res_vals - aux)).argmin()]
                else:
                    # Capacitor
                    new_population[t][i] = cap_vals[(np.abs(cap_vals - aux)).argmin()]
                
            # Evaluation
            new_population[t][num_dimensions] = cost(new_population[t][0:num_dimensions])
            
        # Merge old population (archive) with new one
        merged_pop = np.concatenate([archive, new_population])
        # And sort it again
        merged_pop = merged_pop[merged_pop[:,num_dimensions].argsort()]
        # Store the bests in the archive and update best sol
        archive = merged_pop[:archive_size]
        #best_sol = archive[0][0:num_dimensions]
        best_cost[it] = archive[0][num_dimensions]
        
        # Show iteration info
        # f.write('Iteration %i, best cost found %s\n' % (it, format_e(Decimal(best_cost[it]))))
        # f.write('  best solution: ' + str(best_sol)  + '\n')
        # f.write('  cost best solution: ' + str(best_cost[it])  + '\n')
    
    return archive[0] # Best population and cost

filename = 'results_discretoFijo.txt'
f = open(filename, 'w+')
iterations_per_rval = 50
soft_sols = {}
hard_sols = {}
for R1 in res_vals:
    hard_sols_found = 0
    soft_sols_found = 0
    soft_sols[R1] = set()
    hard_sols[R1] = set()
    f.write("Results for R1 = %i\n" % R1)
    for i in range(0, iterations_per_rval):
        print "Running iter %i for R1=%i" % (i, R1)
        best_sol = mainLoop(R1)
        r2, r3, c4, c5 = best_sol[0], best_sol[1], best_sol[2], best_sol[3]
        sens = get_sol_info(R1, r2, r3, c4, c5)[0]
        isSol = is_sol(R1, r2, r3, c4, c5)
        isSoftSol = is_soft_sol(R1,r2,r3,c4,c5)
        f.write("%i: %s\tSens: %.5f" % (i, str(best_sol), sens))
        if (isSol):
            hard_sols[R1].add((R1,r2,r3,c4,c5))
            hard_sols_found += 1
            f.write(" ES HARD SOLUCION")
        elif (isSoftSol):
            soft_sols[R1].add((R1,r2,r3,c4,c5))
            soft_sols_found += 1
            f.write(" ES SOFT SOLUCION")
        
        f.write("\n")
    f.write("ITERS WITH HARD FOUND SOLS: %i----------------------------\n" % hard_sols_found)
    for h_sol in hard_sols[R1]:
        f.write(str(h_sol) + "\n")
    f.write("ITERS WITH SOFT FOUND SOLS: %i----------------------------\n" % soft_sols_found)
    for s_sol in soft_sols[R1]:
        f.write(str(s_sol) + "\n")
    f.write("---------------------------------------\n")
    