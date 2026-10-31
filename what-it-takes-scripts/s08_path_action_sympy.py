"""Friston (2019), eqs. 1.2-1.4, 1.9, 1.10: Lagrangian, momentum, Hamiltonian and action of a path.
Setting: 2-D state, general surprisal I(x1,x2), constant antisymmetric Q, scalar Gamma, hbar = 1 (eqs. 1.2-1.9);
1-D with Q = 0 for eq. 1.10. Each line prints a difference that should be 0 (or the stated expression)."""
import sympy as sp
x1,x2,v1,v2=sp.symbols('x1 x2 v1 v2',real=True); G,q=sp.symbols('Gamma q',positive=True)
I=sp.Function('I')(x1,x2); X=sp.Matrix([x1,x2]); v=sp.Matrix([v1,v2])
Q=sp.Matrix([[0,q],[-q,0]]); grad=sp.Matrix([I.diff(x1),I.diff(x2)]); lap=I.diff(x1,2)+I.diff(x2,2)
f=(Q-G*sp.eye(2))*grad; divf=f[0].diff(x1)+f[1].diff(x2)
L=sp.Rational(1,2)*(((v-f).T*(v-f))[0]/(2*G)+divf)                      # eq. 1.2
V=(f.T*f)[0]/(4*G)+divf/2                                                 # V/hbar, eq. 1.2
print("1.2  L - [v.v/4G - f.v/2G + V]      =",sp.simplify(L-((v.T*v)[0]/(4*G)-(f.T*v)[0]/(2*G)+V)))
p=sp.Matrix([L.diff(v1),L.diff(v2)])
print("1.3  p - (v-f)/2G                   =",sp.simplify(p-(v-f)/(2*G)).T)
H=(p.T*v)[0]-L
print("1.3  H - (v.v/4G - V)               =",sp.simplify(H-((v.T*v)[0]/(4*G)-V)))
onf={v1:f[0],v2:f[1]}
print("1.4  on v=f: H + L                  =",sp.simplify((H+L).subs(onf)),";  L - div f/2 =",sp.simplify(L.subs(onf)-divf/2))
w=v-Q*grad
L19=sp.Rational(1,2)*((w.T*w)[0]/(2*G)+(v.T*grad)[0]+G*((grad.T*grad)[0]/2-lap))
print("1.9  L(printed) - L(1.2)            =",sp.simplify(sp.expand(L19-L)))
H19=sp.Rational(1,2)*((w.T*w)[0]/(2*G)-G*((grad.T*grad)[0]/2-lap))
d=sp.simplify(sp.expand(H19-H))
print("1.9  Legendre H - H(printed)        =",sp.factor(-d))
print("     ... equals (v-Q grad I).Q grad I/(2G)? ",sp.simplify(-d-((w.T*Q*grad)[0]/(2*G)))==0)
print("     ... vanishes on v=f? ",sp.simplify(d.subs(onf))==0,";  for Q=0? ",sp.simplify(d.subs(q,0))==0)
x,xd=sp.symbols('x xdot',real=True); J=sp.Function('I')(x); f1=-G*J.diff(x)
L1=sp.Rational(1,2)*((xd-f1)**2/(2*G)+f1.diff(x))
V1=G/2*(J.diff(x)**2/2-J.diff(x,2))                                      # V/hbar, eq. 1.10
print("1.10 L - [xd^2/4G + Idot/2 + V]     =",sp.simplify(L1-(xd**2/(4*G)+J.diff(x)*xd/2+V1)))
