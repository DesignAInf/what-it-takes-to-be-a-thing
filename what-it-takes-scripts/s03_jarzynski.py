"""Jarzynski equality for an overdamped particle in a dragged harmonic trap (k_B T = 1).
U(x,t) = k (x - L t/tau)^2 / 2, trap moved a distance L in time tau; Delta F = 0, so E[exp(-w)] should be 1."""
import numpy as np
rng=np.random.default_rng(2)
k,L,tau,dt,N=2.0,2.0,1.0,1e-3,200000
x=rng.standard_normal(N)/np.sqrt(k); w=np.zeros(N); v=L/tau
for i in range(int(tau/dt)):
    c=v*i*dt
    w+= -k*(x-c)*v*dt              # dW = dU/dt dt
    x+= -k*(x-c)*dt+np.sqrt(2*dt)*rng.standard_normal(N)
e=np.exp(-w); print("E[exp(-w)] = %.3f +- %.3f"%(e.mean(),e.std()/np.sqrt(N)))
print("<w> = %.3f ; fraction w<0 = %.3f"%(w.mean(),(w<0).mean()))
r=v**2*tau-(v**2/k)*(1-np.exp(-k*tau)); print("analytic <w> = %.3f"%r)
