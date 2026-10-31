"""Housekeeping heat at NESS (F 6.17 holds only with Q = 0).
2-D Ornstein-Uhlenbeck flow f = (Q - Gamma) H x with Q antisymmetric, Gamma scalar: the density is stationary (Lyapunov check),
the mean surprisal difference between path endpoints is zero, but the mean heat (entropy flow to the medium) per unit time is
E[grad I . Q^T Q grad I]/Gamma = tr(Q^T Q H)/Gamma > 0. Analytic value compared with a Stratonovich simulation."""
import numpy as np
G=0.5; H=np.array([[1.0,0.3],[0.3,2.0]]); Q=np.array([[0,0.8],[-0.8,0]]); J=(Q-G*np.eye(2))@H
S=np.linalg.inv(H); print("Lyapunov residual |J S + S J^T + 2 Gamma I| = %.2e"%np.abs(J@S+S@J.T+2*G*np.eye(2)).max())
rate=np.trace(Q.T@Q@H)/G; print("analytic housekeeping heat rate (units of k_B T per unit time) = %.4f"%rate)
rng=np.random.default_rng(1); n=4000; dt=2e-3; T=20.0; steps=int(T/dt)
x=rng.multivariate_normal(np.zeros(2),S,size=n); x0=x.copy(); q=np.zeros(n)
for _ in range(steps):
    xn=x+(x@J.T)*dt+np.sqrt(2*G*dt)*rng.standard_normal((n,2))
    xm=0.5*(x+xn); q+=np.einsum('ij,ij->i',xm@J.T,xn-x)/G; x=xn
I=lambda y:0.5*np.einsum('ij,jk,ik->i',y,H,y)
print("simulation: mean heat rate = %.3f ; mean surprisal difference I(x0)-I(xT) = %.3f"%(q.mean()/T,(I(x0)-I(x)).mean()))
