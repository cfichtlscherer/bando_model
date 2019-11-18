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


##################### Definition of the constants ##############################


tau = 30         # reaction time
L = 1            # track length
N = 10           # number of vehicles
delta_t = 0.002  # fineness of discretization for solving the ode 
time_end = 10    # how long the model will run
epsilon = L/2    # for the bottleneck model


################### Definition of the extended Bando model #####################

def optimal_vel(h, tau, L):
    """the optimal velocity function returns a velocity for a given headspace"""

    opt_v = tau/L * (np.tanh(h-2) + np.tanh(2)) / (1+np.tanh(2))

    return opt_v


# vecotrize th function
optimal_vel_vec = np.vectorize(optimal_vel)


def optimal_vel_bottleneck(h, tau, L, x, epsilon):
    """the optimal velocity function for the bottleneck model"""
    xi = x%L

    bottleneck_factor = (1- epsilon * np.exp(- (xi - (L/2))**2 ))
    opt_v = optimal_vel(h, tau, L)
    
    opt_v_bottleneck = bottleneck_factor * opt_v

    return opt_v_bottleneck


# vecotrize th function
optimal_vel_bottleneck_vec = np.vectorize(optimal_vel_bottleneck)


def bando_model(x, v, N, tau, alpha, bottleneck, epsilon):
    """ the bando model with the aggressivity term, x and v are vectors 
    describing the position and the speed of every vehicle. Returns the 
    acceleration of the vehicle with number index."""
   
    # ugly, but needed for the last car, with if condition we couldn't vectorize
    xi1, xi2 = x, np.append(x, x+1)[1:N+1]
    vi1, vi2 = v, np.append(v, v)[1:N+1]

    h = xi2 - xi1

    if bottleneck == False:
        bando_term = optimal_vel(h, tau, L) - vi1

    if bottleneck == True:
        bando_term = optimal_vel_bottleneck(h, tau, L, x, epsilon) -vi1

    aggre_term = vi2 - vi1
    
    acceleration = alpha * bando_term + (1-alpha) * aggre_term
    
    return acceleration


# vecotrize the function
bando_model_vec = np.vectorize(bando_model)


################ Definition of the quasistationary solution ####################

def start_values(N, tau, L, random_pertubation = False,
                single_car_pertubation = False, bottleneck = False, epsilon=L/2):
    """
    The start values will be defined in here, including pertubations
    """
    pertubation_x = np.zeros(N)
    pertubation_v = np.zeros(N)

    if random_pertubation == True:
        pertubation_x = np.random.random(N) * 0.25
        pertubation_v = np.random.random(N) * 0.1

    if single_car_pertubation == True:
        pertubation_x[1] = 1/N
        pertubation_v[1] = optimal_vel(5/N, tau, L)

    x = np.arange(N)/N + pertubation_x

    if bottleneck == False:
        v = np.ones(N) * optimal_vel(1/N, tau, L) + pertubation_v

    if bottleneck == True:
        v = np.ones(N) * optimal_vel_bottleneck(1/N, tau, L, x, epsilon) + pertubation_v

    return x, v


########################## Solve the ODE #######################################

def update_xv(x, v, a, delta_t):
    """ updates the position and the speed for the single vehicles"""
    
    x_new = x + delta_t * v
    v_new = v + delta_t * a
    
    return x_new, v_new


# vecotrize the function
update_xv_np = np.vectorize(update_xv)


def solve_ode(x, v, N, tau, alpha, L, delta_t, time_end, bottleneck=False, epsilon=L/2):
    """ solves the ODE and returns x_values, v_values, a_values where the data
    for every step is stored"""

    x_values = x
    v_values = v
    a_values = bando_model_vec(x, v, N, tau, alpha, bottleneck, epsilon)

    for i in np.arange(0, time_end, delta_t):
        a = bando_model_vec(x, v, N, tau, alpha, bottleneck, epsilon)
        x, v = update_xv_np(x, v, a, delta_t)
        
        # round the results
        # x, v, a = np.round(x,5), np.round(v,5), np.round(a,5)

        x_values = np.vstack((x_values, x))
        v_values = np.vstack((v_values, v))
        a_values = np.vstack((a_values, a))
    

    return x_values, v_values, a_values


########################## Plot the results ####################################


def create_the_x_plot(alpha, x_values, delta_t, time_end):
    """ creates the plot"""
    
    for i in range(N):
        plt.plot(np.arange(0, time_end + delta_t, delta_t), x_values[:,i])

    plt.title('Dinstance travelled by the single vehicles, alpha=' + str(alpha))
    plt.grid(b=True, which='both', color='0.9',linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Position')
    filename = str(dt.datetime.now().strftime("%y%m%d_%H:%M:%S__")) + str(alpha) + "_x"
    plt.savefig(filename + ".png")
    plt.close()


def create_the_v_plot(alpha, v_values, delta_t, time_end):
    """ creates the plot"""
    
    for i in range(N):
        plt.plot(np.arange(0, time_end + delta_t, delta_t), v_values[:,i])

    plt.title('Velocity of the single vehicles, alpha=' + str(alpha))
    plt.grid(b=True, which='both', color='0.9',linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    filename = str(dt.datetime.now().strftime("%y%m%d_%H:%M:%S__")) + str(alpha) + "_v"
    plt.savefig(filename + ".png")
    plt.close()


def create_the_a_plot(alpha, a_values, delta_t, time_end):
    """ creates the plot"""
    
    for i in range(N):
        plt.plot(np.arange(0, time_end + delta_t, delta_t), a_values[:,i])

    plt.title('Acceleration of the single vehicles, alpha=' + str(alpha))
    plt.grid(b=True, which='both', color='0.9',linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    filename = str(dt.datetime.now().strftime("%y%m%d_%H:%M:%S__")) + str(alpha) + "_a"
    plt.savefig(filename + ".png")
    plt.close()


########################### Run the code #######################################

# alpha is the aggresivity factor
for alpha in [1]:
    for epsilon in [0.1, 0.5, 0.9]:
        x, v = start_values(N, tau, L, random_pertubation = False, single_car_pertubation = False, bottleneck=True, epsilon=epsilon)
        x_values, v_values, a_values = solve_ode(x, v, N, tau, alpha, L, delta_t, time_end, bottleneck=True, epsilon=epsilon)
        create_the_x_plot(alpha, x_values, delta_t, time_end)
        create_the_v_plot(alpha, v_values, delta_t, time_end)
        create_the_a_plot(alpha, a_values, delta_t, time_end)

