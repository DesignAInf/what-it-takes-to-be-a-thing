"""Friston (2019), eq. 8.25: free energy under the Laplace (Gaussian) assumption, one dimension.
For a Gaussian q = N(mu, s^2) and a quadratic surprisal I(eta, b) = -ln p(eta, b):
F(mu, s) = E_q[I] - H[q]; the optimal variance is s^2 = 1/I''; at the optimum over mu, F = -ln p(b)."""
import sympy as sp
eta,mu,b=sp.symbols('eta mu b',real=True); s,a,c,k=sp.symbols('s alpha c k',positive=True); r=sp.symbols('r',real=True)
# joint Gaussian p(eta,b): eta ~ N(0, 1/a), b | eta ~ N(r eta, 1/c)
logp=sp.log(sp.sqrt(a/(2*sp.pi)))-a*eta**2/2+sp.log(sp.sqrt(c/(2*sp.pi)))-c*(b-r*eta)**2/2
I=-logp; q=sp.exp(-(eta-mu)**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)
F=sp.simplify(sp.integrate(q*I,(eta,-sp.oo,sp.oo))-sp.Rational(1,2)*sp.log(2*sp.pi*sp.E*s**2))
s2opt=sp.solve(sp.diff(F,s),s); print("optimal s^2 =",[sp.simplify(v**2) for v in s2opt],"; 1/I'' =",sp.simplify(1/sp.diff(I,eta,2)))
Fs=F.subs(s,s2opt[0]); muopt=sp.solve(sp.diff(Fs,mu),mu)[0]
pb=sp.integrate(sp.exp(logp),(eta,-sp.oo,sp.oo))
pb=sp.simplify(pb.rewrite(sp.erf))
print("F(optimum) + ln p(b) =",sp.simplify(sp.expand_log(Fs.subs(mu,muopt)+sp.log(pb),force=True)))
