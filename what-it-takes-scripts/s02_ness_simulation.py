"""Euler-Maruyama simulation: solenoidal flow leaves a Gaussian NESS density unchanged.
dx = (Q - Gamma) grad I dt + sqrt(2 Gamma) dW, I = x^T P x / 2, target covariance diag(2.25, 0.64)."""
import numpy as np
rng=np.random.default_rng(1)
Sig=np.diag([2.25,0.64]); P=np.linalg.inv(Sig); G=0.5; q=1.0
Q=np.array([[0,q],[-q,0]]); A=(Q-G*np.eye(2))@P
dt=0.002; n=400000; burn=20000; x=np.zeros(2); X=np.empty((n,2))
noise=rng.standard_normal((n,2))*np.sqrt(2*G*dt)
for t in range(n):
    x=x+A@x*dt+noise[t]; X[t]=x
print("sample covariance:\n",np.round(np.cov(X[burn:].T),3)); print("target:\n",Sig)
