"""Two quantum-mechanical identities used in the book (ch. 7 and the Orientation chapter).
(i) Friston's Schroedinger potential V = (hbar^2/4m)(|I'|^2/2 - I'') equals minus Bohm's quantum potential
    Q = -(hbar^2/2m) (sqrt p)''/sqrt p, with p = exp(-I).
(ii) For psi = sqrt(rho) exp(iS/hbar): Var(p) = hbar^2 F/4 + Var(S'), F = Fisher information of rho.
    Checked for a Gaussian rho with phases S = 0, a x, a x + b x^2."""
import sympy as sp
x=sp.symbols('x',real=True); hb,m,s=sp.symbols('hbar m sigma',positive=True); a,b=sp.symbols('a b',real=True)
I=sp.Function('I')(x); R=sp.exp(-I/2)
V=hb**2/(4*m)*(sp.diff(I,x)**2/2-sp.diff(I,x,2)); Q=-hb**2/(2*m)*sp.diff(R,x,2)/R
print("(i)  V + Q_Bohm =",sp.simplify(V+Q))
rho=sp.exp(-x**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)
F=sp.simplify(sp.integrate(sp.diff(rho,x)**2/rho,(x,-sp.oo,sp.oo)))
for S in [sp.Integer(0),a*x,a*x+b*x**2]:
    psi=sp.sqrt(rho)*sp.exp(sp.I*S/hb)
    E=lambda op: sp.simplify(sp.integrate(sp.conjugate(psi)*op,(x,-sp.oo,sp.oo)))
    var=sp.simplify(E(-hb**2*sp.diff(psi,x,2))-E(-sp.I*hb*sp.diff(psi,x))**2)
    dS=sp.diff(S,x); vS=sp.simplify(sp.integrate(rho*dS**2,(x,-sp.oo,sp.oo))-sp.integrate(rho*dS,(x,-sp.oo,sp.oo))**2)
    print("(ii) S =",S,": Var(p) - hbar^2 F/4 - Var(S') =",sp.simplify(var-hb**2*F/4-vS),"; Var(S') =",vS)
