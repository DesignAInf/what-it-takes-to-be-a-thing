"""Most likely path vs noise-free flow (Onsager-Machlup action, 1-D, constant Gamma).
(i)  symbolic Euler-Lagrange equation of L = (x'-f)^2/(4 Gamma) + f'(x)/2  ->  x'' = f f' + Gamma f''
(ii) OU drift f(x) = -x, x(0)=0, x(1)=1: the minimizer is sinh(t)/sinh(1); the flow from 0 stays at 0
     (action of x_* compared with nearby paths satisfying the same endpoints)."""
import sympy as sp, numpy as np
trap=getattr(np,'trapezoid',None) or np.trapz
t,G=sp.symbols('t Gamma',positive=True); x=sp.Function('x')(t); f=sp.Function('f')
L=(x.diff(t)-f(x))**2/(4*G)+sp.diff(f(x),x)/2
el=sp.euler_equations(L,x,t)[0].lhs
xdd=sp.solve(el,x.diff(t,2))[0]
print("(i)  x'' =",sp.simplify(xdd))
sol=sp.dsolve(sp.Eq(x.diff(t,2),x),x,ics={x.subs(t,0):0,x.subs(t,1):1})
print("(ii) minimizer for f=-x, x(0)=0, x(1)=1 equals sinh(t)/sinh(1):",sp.simplify((sol.rhs-sp.sinh(t)/sp.sinh(1)).rewrite(sp.exp))==0)
tt=np.linspace(0,1,20001); Gam=0.5
def action(xx):
    v=np.gradient(xx,tt); return trap((v+xx)**2/(4*Gam)-0.5,tt)
xstar=np.sinh(tt)/np.sinh(1)
alts=[xstar+0.05*np.sin(np.pi*k*tt) for k in (1,2,3)]+[tt, tt**2]
print("(ii) action of x_* = %.6f ; min action of 5 other paths with the same endpoints = %.6f"%(action(xstar),min(action(a) for a in alts)))
