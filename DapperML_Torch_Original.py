# %%
%load_ext autoreload
%autoreload 2
from plot_helpers import plot_x_y_da, plot_x_3d

import numpy as np
import dapper as dpr
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.Lorenz63 import LPs, Tplot, dstep_dx, step, x0
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from tqdm import tqdm

# %%
dko = 100
tseq = modelling.Chronology(0.01, dko=dko, Ko=1000, Tplot=Tplot, BurnIn=4*Tplot)
Nx = Ny = 3
Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(C=2, mu=x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 2  # modelling.GaussRV(C=CovMat(2*eye(Nx)))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(jj)

dpr.set_seed(3000)#Seed

# %%
HMM.tseq.T = 50
xx, yy = HMM.simulate()

plot_x_y_da(xx,yy)
# %%
xp1 = da.EnKF('Sqrt', N = 500, infl = 1.02, rot = True)

xp1.assimilate(HMM,xx,yy,liveplots = False)

xp1.stats.average_in_time()

# %%
trueVals= np.zeros((yy.shape[0],3))
counter2 = 0
counter =0
testVals = np.zeros((dko,3))
for x in range(xx.shape[0]):
    testVals[counter] = xx[x]
    counter = counter+1
    if ((x+1)%dko == 0):
        trueVals[counter2] = xx[x]
        counter2 = counter2+1
        break
Nx = Ny = 3

# %% Lorenz 63 system manual?
def F(x, sigma=10, beta=8/3, rho=28):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

x0, y0, z0 = 4.0, 10.0, 1.0
initial_conditions = np.array([x0, y0, z0])

Dt = 10
dt = 0.01  # Time step for generating the true trajectory
Ns = 5000 # Number of samples
Nt_shift = 10 # Number of integration time steps between samples
r = 1 # Observation stddev
# Spinup run length and data generating run length
Nt_spinup = 10000
Nt_truth = Ns*Nt_shift+Dt
x_truth = np.empty((Nt_truth,Nx))
x_truth[0] = initial_conditions
x = initial_conditions.copy()  # Initialize x with initial conditions

# %%
with tqdm(total=Nt_truth - 1, desc='L63 trajectory') as progress:
    for i in range(1,Nt_truth):  # Note: -1 because the initial condition is already added
        x += dt * F(x)
        x_truth[i] =x   # Append the new state to the trajectory
        progress.set_postfix_str(x, refresh=False)
        progress.update()

# %%
plot_x_3d(x_truth)

# %%
x_perturbed_all_t = x_truth + np.random.normal(loc=0, scale=r, size=(Nt_truth, Nx))#Used to introduce noise to the trajectory of the L63 system. Does help avoid overfitting.
x_train = x_perturbed_all_t[:Nt_truth-Dt:Nt_shift]#Spaced by Nt_shift. x_perturbed has perturbed trajectory of the L63. Sampled starting from beginning and each sample is separated by a lead time of Dt 
y_train = x_perturbed_all_t[Dt:Nt_truth:Nt_shift]#Represents state of system at future points, shifted by Dt.

# Plot the true and observed trajectories
# plot_x_y_da(x_train,y_train) # The only difference between x and y is that y is shifted by Dt
plot_x_3d(x_train)

# %%
