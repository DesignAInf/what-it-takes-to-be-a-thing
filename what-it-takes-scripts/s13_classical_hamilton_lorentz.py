"""Friston (2019), eqs. 7.7/7.15 and 7.13.
(i) Sign of Q_bb: with b = (a, s) and flow b' = Q grad I, the printed Q = [[0,-1],[1,0]] gives a' = -dI/ds,
    whereas Hamilton's equations a' = +dI/ds, s' = -dI/da need Q = [[0,1],[-1,0]].
(ii) Lorentz force: I(s,a) = z phi(a) + (hbar/2m)(s - z A(a))^2 in 3-D, a' = grad_s I, s' = -grad_a I
    gives m a'' = hbar z (E + a' x B), with E = -grad phi, B = curl A (static potentials)."""
import sympy as sp
a,s=sp.symbols('a s',real=True); I=sp.Function('I')(a,s); gr=sp.Matrix([I.diff(a),I.diff(s)])
for name,Q in [("printed [[0,-1],[1,0]]",sp.Matrix([[0,-1],[1,0]])),("corrected [[0,1],[-1,0]]",sp.Matrix([[0,1],[-1,0]]))]:
    fl=Q*gr; print("(i)  ",name,": a' =",fl[0],", s' =",fl[1])
hb,m=sp.symbols('hbar m',positive=True); z=sp.symbols('z',real=True)
A_=sp.symbols('a1:4',real=True); S_=sp.symbols('s1:4',real=True); a_=sp.Matrix(A_); s_=sp.Matrix(S_)
phi=sp.Function('phi')(*A_); Av=sp.Matrix([sp.Function('A%d'%i)(*A_) for i in (1,2,3)])
I3=z*phi+hb/(2*m)*((s_-z*Av).T*(s_-z*Av))[0]
adot=sp.Matrix([I3.diff(si) for si in S_]); sdot=sp.Matrix([-I3.diff(ai) for ai in A_])
jac=lambda vec: sp.Matrix([[vec[i].diff(A_[j]) for j in range(3)] for i in range(3)])
addot=jac(adot)*adot+sp.Matrix([[adot[i].diff(S_[j]) for j in range(3)] for i in range(3)])*sdot   # chain rule
E=-sp.Matrix([phi.diff(ai) for ai in A_]); B=sp.Matrix([Av[2].diff(A_[1])-Av[1].diff(A_[2]),Av[0].diff(A_[2])-Av[2].diff(A_[0]),Av[1].diff(A_[0])-Av[0].diff(A_[1])])
print("(ii)  m a'' - hbar z (E + a' x B) =",sp.simplify(m*addot-hb*z*(E+adot.cross(B))).T)
