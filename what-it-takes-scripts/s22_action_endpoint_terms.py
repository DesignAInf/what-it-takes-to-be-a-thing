"""s22. F 10.7 (Friston 2019, p. 113): can the integral terms of the action be dropped for small noise?
Onsager-Machlup action for a gradient flow f = -Gamma * grad I (Q = 0, constant scalar Gamma):
    A = 1/2 [I(x_T) - I(x_0)] + int (|xdot|^2 + |f|^2)/(4 Gamma) dt + 1/2 int div f dt.
(i)   Along the noise-free flow the kinetic term equals -Delta I / 4, whatever Gamma: half the endpoint term.
(ii)  For any path, int (|xdot|^2+|f|^2)/(4 Gamma) >= |Delta I|/2, so S := A - divergence term >= max(Delta I, 0).
(iii) The bound is not attained for arbitrary endpoints. OU process dX = -X dt + sqrt(2 Gamma) dW, I(x) = x^2/(2 Gamma).
      For fixed duration T the minimal cost S(T) = (1/4Gamma) min int (xdot + x)^2 dt has a closed form
      (Euler-Lagrange: xddot = x). We minimize S(T) over T (bounded scalar minimization, convergence checked, plus a
      dense grid and the T -> infinity limit) and compare with the analytic quasi-potential of the 1-D double-sided well:
      0 if b lies between 0 and a (follow the flow); I(b) - I(a) if a lies between 0 and b (time-reversed flow);
      I(b) - I(0) if 0 lies strictly between a and b (descend to 0, then climb).  Values are reported as Gamma * S."""
import numpy as np
from scipy.optimize import minimize_scalar
trap=getattr(np,'trapezoid',None) or np.trapz
# (i) and (ii): I(x) = x^2/2, f = -Gamma x
I=lambda x: x**2/2
for Gam in (1.0,0.1,0.01):
    T=5/Gam; t=np.linspace(0,T,200001); x=2*np.exp(-Gam*t); xd=-Gam*x
    K=trap(xd**2/(4*Gam),t); dI=I(x[-1])-I(x[0])
    print("(i)   Gamma=%-5g kinetic = %.6f ; -Delta I/4 = %.6f ; endpoint term Delta I/2 = %.6f"%(Gam,K,-dI/4,dI/2))
rng=np.random.default_rng(0); Gam=0.5; worst=np.inf
for k in range(200):
    T=rng.uniform(.5,5); t=np.linspace(0,T,4001); c=rng.normal(size=4)
    x=c[0]+c[1]*t+c[2]*np.sin(t)+c[3]*t**2/T; xd=np.gradient(x,t); f=-Gam*x
    lhs=trap((xd**2+f**2)/(4*Gam),t); rhs=abs(I(x[-1])-I(x[0]))/2; worst=min(worst,lhs-rhs)
print("(ii)  200 random paths: min of [integral terms - |Delta I|/2] = %.4f (>= 0)"%worst)
# (iii) closed form at fixed T (in units Gamma*S): x = A e^t + B e^-t, xdot + x = 2 A e^t
def GS(T,a,b):
    A=(b-a*np.exp(-T))/(2*np.sinh(T))
    return A**2*(np.exp(2*T)-1)/2          # Gamma * (1/4Gamma) int (2A e^t)^2 dt
def V(a,b):                                   # analytic quasi-potential, times Gamma; I = x^2/(2 Gamma)
    Ig=lambda x: x**2/2
    if a*b>0 and abs(b)<=abs(a): return 0.0
    if a*b>0 or a==0: return Ig(b)-Ig(a)
    return Ig(b)
print("(iii) a -> b : Gamma*Delta I | Gamma*max(Delta I,0) | analytic quasi-potential | infimum over T of the closed-form S(T)")
for a,b in ((0.5,2.0),(2.0,0.5),(1.0,-0.5),(-0.3,1.2)):
    grid=np.linspace(1e-3,40,400001); vals=GS(grid,a,b); k=int(np.argmin(vals)); lim=b**2/2; dI=(b**2-a**2)/2
    if k<len(grid)-1 and vals[k]<lim-1e-9:    # interior minimum below the T -> infinity limit: refine, check convergence
        r=minimize_scalar(GS,bounds=(grid[max(k-1,0)],grid[k+1]),args=(a,b),method="bounded",options={"xatol":1e-12})
        res="%.6f at T* = %.4f (%s)"%(r.fun,r.x,"converged" if r.success else "NOT CONVERGED")
    else:                                     # S(T) decreasing on the grid: infimum is the T -> infinity limit
        res="%.6f = limit T -> inf, approached but not attained: S(5), S(10), S(20) = %.6f, %.8f, %.10f"%(lim,GS(5,a,b),GS(10,a,b),GS(20,a,b))
    print("      %4.1f -> %4.1f : %8.4f | %8.4f | %8.4f | %s"%(a,b,dI,max(dI,0),V(a,b),res))
print("      1 -> -1/2: Delta I = -3/(8 Gamma) < 0, yet the least cost is 1/(8 Gamma) > 0 (descend to 0, then climb).")
