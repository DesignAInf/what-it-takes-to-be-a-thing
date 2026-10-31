"""Biehl, Pollock & Kanai (2021), examples A4 (Appendix A) and A11 (Appendix B) (linear Gaussian, Gamma = I, order (eta, s, a, lambda)).
Steady-state covariance S from M S + S M' + 2 Gamma = 0, precision U = inv(S), solenoidal R = -M S - Gamma.
A4: flow condition without a statistical Markov blanket: U_{eta,lambda} = -12/85 != 0 (8 R as published; 255 U as published except the (1,3) entry, printed -31/85 in BPK (A6)).
A11: blanket holds (U_{eta,lambda} = 0, 1057 U_{eta,eta} = 1023) but the flow of lambda differs from the free-energy form (eq. 20).
The first block uses floating-point linear algebra (numpy/scipy); the last block repeats the A4 and A11 precision checks exactly with sympy."""
import numpy as np, scipy.linalg as sla
ness=lambda M: sla.solve_continuous_lyapunov(M,-2*np.eye(len(M)))
c=-2/3; M=np.array([[-1,c,c,0],[c,-1,c,0],[0,c,-1,c],[0,c,c,-1]],float)
S=ness(M); U=np.linalg.inv(S); R=-M@S-np.eye(4)
print("A4  255*U =",np.round(255*U,3).tolist()); print("A4  8*R =",np.round(8*R,4).tolist()); print("A4  U_eta,lambda =",U[0,3]," (-12/85 =",-12/85,")")
r=1/(2*np.sqrt(2)); M2=np.array([[-1,-r,-r,0],[-r,-1,-1/16,0],[0,1/16,-1,-r],[0,r,-r,-1]])
S2=ness(M2); U2=np.linalg.inv(S2); R2=-M2@S2-np.eye(4)
print("A11 U_eta,lambda =",U2[0,3],"; 1057*U_eta,eta =",1057*U2[0,0])
Usal=np.linalg.inv(S2[1:,1:])
print("A11 true flow of lambda on (s,a,lambda):",M2[3,1:].round(6).tolist())
print("A11 free-energy form (eq. 20):           ",(-(1+R2[3,3])*Usal[2,:]).round(6).tolist())

# Exact check with sympy (rational arithmetic for A4, exact algebraic numbers for A11)
import sympy as sp
def ness_exact(Mx):
    n=Mx.shape[0]; sy=sp.symbols('s0:%d'%(n*(n+1)//2)); Sx=sp.zeros(n); k=0
    for i in range(n):
        for j in range(i,n): Sx[i,j]=Sx[j,i]=sy[k]; k+=1
    eqs=list(Mx*Sx+Sx*Mx.T+2*sp.eye(n)); sol=sp.solve(eqs,sy,dict=True)[0]
    return Sx.subs(sol)
c=sp.Rational(-2,3); Me=sp.Matrix([[-1,c,c,0],[c,-1,c,0],[0,c,-1,c],[0,c,c,-1]])
Se=ness_exact(Me); Ue=sp.simplify(Se.inv()); Re=sp.simplify(-Me*Se-sp.eye(4))
print("A4  exact: 255*U =",(255*Ue).tolist()); print("A4  exact: 8*R =",(8*Re).tolist()); print("A4  exact: U_eta,lambda =",Ue[0,3])
r=1/(2*sp.sqrt(2)); Me2=sp.Matrix([[-1,-r,-r,0],[-r,-1,-sp.Rational(1,16),0],[0,sp.Rational(1,16),-1,-r],[0,r,-r,-1]])
Se2=ness_exact(Me2); print("A11 exact: R =",sp.simplify(-Me2*Se2-sp.eye(4)).tolist())
Ue2=sp.simplify(Se2.inv()); print("A11 exact: U_eta,lambda =",sp.simplify(Ue2[0,3]),"; 1057*U_eta,eta =",sp.nsimplify(sp.simplify(1057*Ue2[0,0])))
