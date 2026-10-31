"""Friston (2019), eqs. 1.15-1.18 and 2.10: Fisher information, KL divergence and information length.
(i) Gaussian family N(m, s^2): the Hessian of KL in m equals the Fisher information 1/s^2.
(ii) a Gaussian whose mean relaxes as m0 exp(-g t) at fixed width s (the OU solution when s^2 = Gamma/gamma, g = gamma): m(t) = m0 exp(-g t);
     the time-Fisher information int pdot^2/p dx = mdot^2/s^2 and the information length l(inf) = m0/s.
(iii) eq. 2.10: with the stationary width s^2 = Gamma/gamma, l(inf) = |pi0| sqrt(gamma/Gamma)."""
import sympy as sp
x,m,m2=sp.symbols('x m m2',real=True); s,t,g,m0,Gam=sp.symbols('sigma t gamma m0 Gamma',positive=True)
p=lambda mu: sp.exp(-(x-mu)**2/(2*s**2))/sp.sqrt(2*sp.pi*s**2)
KL=sp.simplify(sp.integrate(p(m)*(sp.log(p(m))-sp.log(p(m2))),(x,-sp.oo,sp.oo)))
fisher=sp.simplify(sp.integrate(p(m)*sp.diff(sp.log(p(m)),m)**2,(x,-sp.oo,sp.oo)))
print("(i)   KL =",KL,"; d2KL/dm2^2 at m2=m:",sp.simplify(sp.diff(KL,m2,2).subs(m2,m)),"; Fisher:",fisher)
mt=m0*sp.exp(-g*t); pt=p(mt)
tf=sp.simplify(sp.integrate(sp.diff(pt,t)**2/pt,(x,-sp.oo,sp.oo)))
print("(ii)  time-Fisher - mdot^2/s^2 =",sp.simplify(tf-sp.diff(mt,t)**2/s**2))
l=sp.integrate(sp.sqrt(sp.simplify(tf)),(t,0,sp.oo))
print("(ii)  l(inf) =",sp.simplify(l))
print("(iii) l(inf) with s^2=Gamma/gamma:",sp.simplify(l.subs(s,sp.sqrt(Gam/g))))
