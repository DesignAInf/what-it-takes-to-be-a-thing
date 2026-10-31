"""Steady-state covariance for the 2-D example of Aguilera et al. (2022, App. C).
Drift J = [[-1, 1.2], [-1, -1]]: solve the Lyapunov equation J S + S J^T + 2 Gamma = 0.
With Gamma = I this gives Sigma_yy = 1.0545, Sigma_bb = 0.9545, Sigma_yb = 0.0455 (see the derivation note in Ch. 15).
It also reproduces the simulation of the figure 'Average flows do not track behavior' (Ch. 15): mode m_y(b) = K b with K = Sigma_yb/Sigma_bb, average flow <f_y>_b = c b,
one Euler-Maruyama trajectory (seed 7, dt = 1e-3, T = 30, start (2, 0)), and the ranges of the actual mode and of the
variable integrated along the average flow."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap
J=np.array([[-1,1.2],[-1,-1.0]])
for g in (0.5,1.0):
    S=lyap(J,-2*g*np.eye(2)); print("Gamma=%.1f  Sigma ="%g,np.round(S,4).tolist())
S=lyap(J,-2*np.eye(2)); K=S[0,1]/S[1,1]; c=J[0,0]*K+J[0,1]
print("K = %.4f ; <f_y>_b = %.4f b ; ratio in terms of the mode = %.1f"%(K,c,J[0,0]+J[0,1]*S[1,1]/S[0,1]))
rng=np.random.default_rng(7); dt=1e-3; N=int(30/dt); z=np.zeros((N,2)); z[0]=[2.0,0.0]
for t in range(N-1): z[t+1]=z[t]+J@z[t]*dt+np.sqrt(2*dt)*rng.standard_normal(2)
my=K*z[:,1]; mt=np.empty(N); mt[0]=my[0]
for t in range(N-1): mt[t+1]=mt[t]+c*z[t,1]*dt
print("simulation: actual mode in [%.2f, %.2f] ; integrated average flow in [%.1f, %.1f]"%(my.min(),my.max(),mt.min(),mt.max()))
