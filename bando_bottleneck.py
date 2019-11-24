"""
Nov 13, 2019
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

An implementation of the Bando model with an implmentation of the bottle neck.
Developed during the course 'traffic flow modelling' by Prof. Ingenuin Gasser 
and Hannes von Allwörden at the University of Hamburg.
""" 

import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt

################ Definition of the Bando model with bottleneck #################

def get_headspaces(x, L):
    """calculates the headspaces from the positions of the cars and the length 
    of the road L"""
    
    next_car = np.roll(x, -1) 
    next_car[-1] += L
    h = abs(x - next_car)

    return h


def optimal_vel(h):
    """the optimal velocity function returns a velocity for a given headspace"""

    opt_v = np.tanh(h-2) + np.tanh(2)

    return opt_v


def optimal_vel_bottleneck(x, epsilon, L):
    """the optimal velocity function for the bottleneck model"""

    h = get_headspaces(x, L)

    bottleneck_factor = 1 - (epsilon * np.exp(- ((L/2) - (x%L))**2 ))
    opt_v = optimal_vel(h)
    
    opt_v_bottleneck = bottleneck_factor * opt_v

    return opt_v_bottleneck


def update(x, v, a, dt, L, epsilon):
    """calculates the new positions, velocities and accelerations after dt"""

    x_update = x + dt * v
    v_update = v + dt * a
    a_update = optimal_vel_bottleneck(x, epsilon, L) - v

    return x_update, v_update, a_update


########################## Solve the ODE #######################################

def solve_ode(x_start, v_start, a_start, L, epsilon, dt, time_end):
    """ solves the ODE and returns x_values, v_values, a_values where the data
    for every step is stored"""

    x_update, v_update, a_update = x_start, v_start, a_start
    x_values, v_values, a_values = x_start, v_start, a_start

    for i in np.arange(0, time_end, dt):
        
        x_update, v_update, a_update = update(x_update, v_update, a_update, dt, L, epsilon)

        x_values = np.vstack((x_values, x_update))
        v_values = np.vstack((v_values, v_update))
        a_values = np.vstack((a_values, a_update))

    return x_values, v_values, a_values


########################## Plot the results ####################################

def create_the_plot(values, dt, plot_title="title", yaxis="y"):
    """ creates the plot"""
    
    for i in range(len(values[0])):
        plt.plot(np.arange(len(values)) * dt, values[:,i])

    plt.title(plot_title)

    plt.grid(b=True, which='both', color='0.9',linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(yaxis)
    
    plt.show()

########################### Run the code #######################################

L = 20
NC = 10
epsilon = 0.2
dt = 0.005
time_end = 200

x_start = np.arange(NC) * L/NC + np.random.random(NC)
v_start = np.ones(NC) * optimal_vel(L/NC) 
a_start = np.zeros(NC) + np.random.random(NC)

xre, vre, are = solve_ode(x_start, v_start, a_start, L, epsilon, dt, time_end)
plt.title("N = 10, L = 20, epsilon = 0.2")
plt.xlabel("x")
plt.ylabel("v")
#plt.plot(np.round(xre[:,1:5], 5) , np.round(vre[:,1:5], 5), ',')
plt.plot(np.round(xre[:,1], 5) , np.round(vre[:,1], 5), ',')

#x = np.arange(0, L, 0.1)
#plt.plot(x, (1 - epsilon * np.exp(- ((L/2) - (x%L))**2 )) , ',', color = 'red')

plt.show()
#create_the_plot(np.round(vre, 6), dt)
