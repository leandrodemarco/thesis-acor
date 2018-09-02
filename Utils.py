#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 18:16:27 2018

@author: leandrodemarcovedelago
"""
import numpy as np
import math

# ---------------------- Filter constants ----------------------
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

# ------------- Desired values for gain, omega and Q -------------
gobj = 3.0
wp = 1000*2*math.pi
Qp = 1/math.sqrt(2.0)
invQp=1/Qp

# ---------------------- Auxiliar functions ----------------------
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

def get_sol_info(r1, r2, r3, c4, c5):
    a = r1/r2
    b = r1/r3
    g = 1/a
    sens = (2 + abs(1-a+b) + abs(1+a-b))/(2*(1+a+b))
    omega = math.sqrt(a*b/(c4*c5))/r1
    invq = math.sqrt(c5/(c4*a*b))*(1+a+b)
    
    return (sens, g, 1./invq, omega)

def is_sol(r1, r2, r3, c4, c5):
    sens, g, q, omega = get_sol_info(r1,r2,r3,c4,c5)
    sensOk = sens < 1
    err = 0.025
    gmax=(1+err)*gobj
    gmin=(1-err)*gobj
    wmax=(1+err)*wp
    wmin=(1-err)*wp
    Qmax=(1+err)*Qp
    Qmin=(1-err)*Qp
    
    gOk = g < gmax and g > gmin
    omegaOk= omega < wmax and omega > wmin
    qOk = q < Qmax and q > Qmin
    
#    if (not gOk):
#        print "Viola ganancia!"
#        print gmin, g, gmax
#    if (not omegaOk):
#        print "Viola omega"
#        print wmin, omega, wmax
#    if (not qOk):
#        print "Viola Q"
#        print Qmin, q, Qmax
#    if (not sensOk):
#        print sens
#        print "Viola sensibilidad"
        
    
    allOk = gOk and omegaOk and qOk and sensOk
    return allOk
#    if (allOk):
#        print sens, g, q, omega
    
def is_soft_sol(r1, r2, r3, c4, c5):
    sens, g, q, omega = get_sol_info(r1,r2,r3,c4,c5)
    err = 0.025
    gmax=(1+err)*gobj
    gmin=(1-err)*gobj
    wmax=(1+err)*wp
    wmin=(1-err)*wp
    Qmax=(1+err)*Qp
    Qmin=(1-err)*Qp
    
    gOk = g < gmax and g > gmin
    omegaOk= omega < wmax and omega > wmin
    qOk = q < Qmax and q > Qmin
    
    return gOk and omegaOk and qOk

# END AUXILIAR FUNCTIONS