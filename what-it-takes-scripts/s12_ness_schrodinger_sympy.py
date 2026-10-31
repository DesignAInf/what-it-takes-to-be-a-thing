"""Friston (2019), eqs. 5.3-5.5: wave function of a NESS density in one dimension.
Q = 0, Gamma = hbar/2m, f = -Gamma I', Psi = sqrt(p) = exp(-I/2).
5.3: Psi' = (m/hbar) f Psi.  5.4: V = (hbar/4Gamma) f^2 + (hbar/2) f' = (hbar^2/4m)(I'^2/2 - I'').
5.5: -(hbar^2/2m) Psi'' + V Psi = 0, i.e. H Psi = 0 (the E = 0 ground state of the potential built from p)."""
import sympy as sp
x=sp.symbols('x',real=True); hb,m=sp.symbols('hbar m',positive=True)
I=sp.Function('I')(x); G=hb/(2*m); f=-G*I.diff(x); Psi=sp.exp(-I/2)
print("5.3  Psi' - (m/hbar) f Psi =",sp.simplify(Psi.diff(x)-(m/hb)*f*Psi))
V=hb/(4*G)*f**2+hb/2*f.diff(x)
print("5.4  V - (hbar^2/4m)(I'^2/2 - I'') =",sp.simplify(V-hb**2/(4*m)*(I.diff(x)**2/2-I.diff(x,2))))
print("5.5  H Psi =",sp.simplify(-hb**2/(2*m)*Psi.diff(x,2)+V*Psi))
