import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # system parameter values-- can be iterated upon
# L=0.3810 # length of pendulum in meters
# m=0.59787 # arm mass in kg
# M=np.array([0,0.5,1,1.5]) # mass added on top in kg
# k_spring=np.array([20.492,31.446,42,50.446,61,80])*0.11298/(np.pi/2) #total spring constant in N-m/rad
# g=9.81 #gravity acceleration in ms^-2
# print(k_spring)

# I_arm=1/3*m*L**2  #moment of inertia
# print(f' moment of inertia for the arm: {I_arm}')

class SpringInvertedPendulum:
    def __init__(self,M,k_spring,b):
        self.g=9.81 #gravity acceleration in m/s^2
        self.m=0.59787 #arm mass in kg
        self.L=0.3810 # arm length in meters
        self.M=M #masses being placed on top
        self.k_spring=k_spring #spring placed on top
        self.I=1/3*self.m*self.L**2+self.M*self.L**2 #moment of inertia
        self.b=2*np.sqrt((self.k_spring-0.5*self.m*self.g*self.L-self.M*self.g*self.L)*self.I)*(1+0.3*np.random.rand()) #adding random ness to the critical damping for better physics randomization
        self.p=self.m*self.L/2+self.M*self.L 
        self.dt=0.01 #time interval (in seconds)
        self.t=0 #initial time

    def physics(self,theta,t_motor,theta_dot):
        theta_ddot=(self.p*self.g*np.sin(theta)+t_motor-self.k_spring*theta-self.b*theta_dot)/self.I #governing equation of the system, found through Euler-Lagrangian method
        theta_dot=theta_dot+theta_ddot*self.dt
        theta=theta+theta_dot*self.dt

        return theta_dot,theta



