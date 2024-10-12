# THIS SCRIPT SOLVES THE FRACTIONAL DIFFERENTIAL SYSTEM CONSIDERED BY CUSSEDDU AND FERREIRA 2024

# THIS CODE SOLVES THE EQUATION 
# D_C^GAMMA I = BETA*(N-I)*I - NU*D_RL^(GAMMA-ALPHA) I
# BY USING THE L1 METHOD, TEXTBOOK BY CHANGPIN LI AND MIN CAI
# Theory and Numerical Approximations of Fractional Integrals and Derivatives

# D_C^GAMMA I IS THE CAPUTO FRACTIONAL DERIVATIVE OF ORDER GAMMA OF THE FUNCTION I
# D_RL^(GAMMA-ALPHA) I IS THE RIEMANN--LIOUVILLE FRACTIONAL DERIVATIVE OF ORDER GAMMA-ALPHA OF THE FUNCTION I

# FOR FURTHER DETAILS SEE APPENDIX A IN CUSSEDDU AND FERREIRA 2024

import numpy as np
from math import gamma
import matplotlib.pyplot as plt


## Definition of the parameters 
gama = 1            # Caputo differentiation order
alpha = 0.95        # gama - alpha is the Riemann--Lioville differentiation order
beta = 0.02     
nu = 1**(-alpha) 
N = 100          

T = 100           # Final time of the simulation
h = 0.05            # Timestep for the temporal discretisation

I0 = 2              # Initial condition



## Check that the condition 0<alpha<=gamma<=1 is satisfied
while gama<alpha or gama > 1 or alpha<=0:
    print('The parameters alpha and gamma need to satisfy 0<alpha<=gamma<=1')
    alpha = float(input('alpha = ')) 
    gama = float(input('gamma = '))

## Check that the initial condition satisfies 0<=I0<=N
while I0>N or I0<0:
    print('The initial condition needs to satisfy 0<=I0<=' + str(N))
    I0 = float(input('I0 = '))


### definition of the solver and plot functions:

def solver(gama, alpha, beta, nu, N, I0, T, h):             # solver of the fractional differential equation

    ### initialisation
    time = np.arange(0*h,T,h)                               # discretise time homogeneously
    I = np.zeros(len(time))                                 # initialise solution vector I
    I[0] = I0                                               # initial condition

    b0_gama = h**(-gama)/gamma(2-gama)                      # coefficient for the L1 discretisation
    b0_gamaalpha = h**(-gama+alpha)/gamma(2-gama+alpha)     # coefficient for the L1 discretisation

    # iterative solutions for all timepoints
    for j in range(1,len(time)):
    
        Ij = beta*(N-I[j-1])*I[j-1] - nu*I[0]*(j*h)**(alpha-gama)/gamma(1-gama+alpha)
        for k in range(j-1):
            b_gama = b0_gama*((j-k)**(1-gama) - (j-k-1)**(1-gama))
            b_gamaalpha = b0_gamaalpha*((j-k)**(1-gama+alpha) - (j-k-1)**(1-gama+alpha))
            Ij -= (b_gama + nu*b_gamaalpha)*(I[k+1] - I[k]) 
        
        I[j] = Ij/(b0_gama + nu*b0_gamaalpha) + I[j-1]

    return time, I

    

def plot_figure(time, I, S):
    plt.rc('font', size=22)  
    plt.figure(figsize=(10,8))
    plt.plot(time,I, linewidth = 2, label = 'I(t)', color = 'red')
    plt.plot(time,S, linewidth = 2, label = 'S(t)', color = 'green')
    plt.xlabel('$t$')
    plt.ylabel('$I(t), S(t)$', rotation=90)
    plt.grid()
    plt.legend()
    plt.savefig('solution_I_and_S.png')



time, I = solver(gama, alpha, beta, nu, N, I0, T, h)   # solve the fractional differential equation
S = N - I                                               # calculate I

plot_figure(time, I, S)                                 # plot and export the figure


