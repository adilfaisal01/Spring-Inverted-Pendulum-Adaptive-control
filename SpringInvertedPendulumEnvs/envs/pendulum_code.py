import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box

# # system parameter values-- can be iterated upon
# L=0.3810 # length of pendulum in meters
# m=0.59787 # arm mass in kg
# M=np.array([0,0.5,1,1.5]) # mass added on top in kg
# k_spring=np.array([20.492,31.446,42,50.446,61,80])*0.11298/(np.pi/2) #total spring constant in N-m/rad
# g=9.81 #gravity acceleration in ms^-2
# print(k_spring)

# I_arm=1/3*m*L**2  #moment of inertia
# print(f' moment of inertia for the arm: {I_arm}')

class SpringInvertedPendulum(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,M=0,k_spring=5.70,max_steps=500,render_mode: str | None=None):
        self.g=9.81 #gravity acceleration in m/s^2
        self.m=0.59787 #arm mass in kg
        self.L=0.3810 # arm length in meters
        self.M=M #masses being placed on top
        self.k_spring=k_spring #spring placed on top
        self.I=1/3*self.m*self.L**2+self.M*self.L**2 #moment of inertia
        self.inner_arg=self.k_spring-0.5*self.m*self.g*self.L-self.M*self.g*self.L

        if self.inner_arg<0:
            print(f'physically impossible pendulum, inner_argument={self.inner_arg}')
            return
        elif self.inner_arg==0:
            print('zero damping')
        else:
            print('good to run')
        self.b=2*np.sqrt((self.inner_arg)*self.I) #adding random ness to the critical damping for better physics randomization
        self.p=self.m*self.L/2+self.M*self.L 
        self.dt=0.01 #time interval (in seconds)
        self.t=0 #initial time


        # action space, defined as [-1.0,1.0] nm from the motor
        self.action_space=Box(low=-2.0,high=2.0,shape=(1,),dtype=np.float64)
        # state space--> angular velocity and angle of the pendulum arm
        high_obs=np.array([np.pi/2, np.inf])
        self.observation_space=Box(low=-high_obs,high=high_obs, shape=(2,),dtype=np.float64)
        # initialize the physics
        self.state: np.array=np.zeros(shape=(2,))
        self.terminated=False
        self.maxsteps=max_steps
        self.current_steps=0

    # # def physics(self,theta,t_motor,theta_dot):
    # #     theta_ddot=(self.p*self.g*np.sin(theta)+t_motor-self.k_spring*theta-self.b*theta_dot)/self.I #governing equation of the system, found through Euler-Lagrangian method
    # #     theta_dot=theta_dot+theta_ddot*self.dt
    # #     theta=theta+theta_dot*self.dt

    #     return theta_dot,theta
    def step(self, action):

        theta,theta_dot=self.state #previous state
        action=np.clip(action,-2,+2)
        t_motor= float(np.ravel(action)[0]) #action taken
        
        #move the physics forward by dt
        theta_ddot=(self.p*self.g*np.sin(theta)+t_motor-self.k_spring*theta-self.b*theta_dot)/self.I #governing equation of the system, found through Euler-Lagrangian method
        theta_dot=theta_dot+theta_ddot*self.dt
        theta=theta+theta_dot*self.dt
        self.state=np.array([theta,theta_dot]) #updated states
        # check for termination
        termination=bool(np.abs(theta)>=np.pi/2)
        if termination:
            self.terminated=True #arm moved too far, get wreckt lol
        else:
            self.terminated=False
        reward=-np.abs(theta) - 0.1 * theta_dot**2 #reward function
        self.current_steps=self.current_steps+1
        if self.current_steps>=self.maxsteps:
            truncated=True
        else:
            truncated=False
        return np.array(self.state,dtype=np.float64),reward,termination,truncated,{'t_motor': t_motor}
    
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.state=self.np_random.uniform(low=np.array([-np.pi/2,-1]),high=np.array([1,np.pi/2]),size=(2,))
        self.terminated=False
        self.current_steps=0
        return np.array(self.state,dtype=np.float64),{}
    def render(self, mode="human"):
        None



        
    



       



