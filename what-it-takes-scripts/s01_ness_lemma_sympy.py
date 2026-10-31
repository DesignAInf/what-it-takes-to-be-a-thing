"""Symbolic checks of the NESS lemma (Friston 2019, eq. 1.8) in 2-D.
Flow f = (Q - Gamma) grad I with constant antisymmetric Q, scalar Gamma, p = exp(-I).
Checks: (i) p is stationary for the Fokker-Planck equation; (ii) the solenoidal current
p Q grad I is divergence-free; (iii) div f = -Gamma lap I + (terms that vanish for constant Q)."""
import sympy as sp
x,y,G,q=sp.symbols('x y Gamma q',real=True)
I=sp.Function('I')(x,y); p=sp.exp(-I)
Q=sp.Matrix([[0,q],[-q,0]]); Gm=G*sp.eye(2)
grad=lambda h: sp.Matrix([sp.diff(h,x),sp.diff(h,y)])
div=lambda v: sp.diff(v[0],x)+sp.diff(v[1],y)
f=(Q-Gm)*grad(I)
fp_rhs=-div(f*p)+G*(sp.diff(p,x,2)+sp.diff(p,y,2))
print("(i)  Fokker-Planck stationarity residual:",sp.simplify(fp_rhs))
print("(ii) div(p Q grad I):",sp.simplify(div(p*Q*grad(I))))
print("(iii) div f + Gamma lap I:",sp.simplify(div(f)+G*(sp.diff(I,x,2)+sp.diff(I,y,2))))
