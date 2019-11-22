"""
Nov 18, 2019
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

An implementation of the Bando model with with several additional features.
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

    bottleneck_factor = 1 - epsilon * np.exp(- ((L/2) - (x%L))**2 )
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

x_start = np.arange(10)
v_start = np.ones(10) * 0.05
a_start = np.ones(10) * 0.02
L = 15
epsilon = 0.5
dt = 0.01
time_end = 100

xre, vre, are = solve_ode(x_start, v_start, a_start, L, epsilon, dt, time_end)

create_the_plot(xre, dt)
