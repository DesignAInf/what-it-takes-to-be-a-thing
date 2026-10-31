"""Aguilera, Millidge, Tschantz & Buckley (2022): weak-coupling expansions (eqs. 28-29, 30-33, 47-48).
Linear system J = -I + C with couplings C of size eps among blocks y (external), s, a, x (internal), Gamma = I,
with the direct couplings y<-x, s<-x, a<-y, x<-y removed. Checks, as eps -> 0:
(28-29) the blanket term H_yx of the precision matches -1/4 (C_ys C_xs' + C_ya C_xa') up to O(eps^3);
(30-33) Q = (C - C')/2 to first order, so Q_ya = (varsigma^2/2) C_ya (varsigma^2 = 1, C_ay = 0); (47-48) the first-order drift formulas.
Random numbers follow the same sequence as the original check (seed 1)."""
import numpy as np, scipy.linalg as sla
ness=lambda M: sla.solve_continuous_lyapunov(M,-2*np.eye(len(M)))
rng=np.random.default_rng(1); idx={'y':[0,1],'s':[2],'a':[3],'x':[4,5]}; n=6
allowed=np.ones((n,n),bool)
for r_,c_ in [('y','x'),('s','x'),('a','y'),('x','y')]:
    for i in idx[r_]:
        for j in idx[c_]: allowed[i,j]=False
Bk=lambda Mx,r_,c_: Mx[np.ix_(idx[r_],idx[c_])]
C0=rng.choice([-1,1],size=(n,n))*allowed; np.fill_diagonal(C0,0)
_=rng.standard_normal((n,n))                       # consumed by an intermediate check in the original session
C0=rng.standard_normal((n,n))*allowed; np.fill_diagonal(C0,0)
b=idx['s']+idx['a']
for eps in [0.04,0.02,0.01]:
    C=eps*C0; J=-np.eye(n)+C; S=ness(J); H=np.linalg.inv(S)
    pred=-0.25*(Bk(C,'y','s')@Bk(C,'x','s').T+Bk(C,'y','a')@Bk(C,'x','a').T)
    Q=np.eye(n)+J@S                                # from J = (Q - Gamma) H with Gamma = I
    Sbb=S[np.ix_(b,b)]; K=S[np.ix_(idx['y'],b)]@np.linalg.inv(Sbb)
    drift=K@(J[np.ix_(b,b)]+J[np.ix_(b,idx['y'])]@K+J[np.ix_(b,idx['x'])]@(S[np.ix_(idx['x'],b)]@np.linalg.inv(Sbb)))
    marg=J[np.ix_(idx['y'],idx['y'])]@K+J[np.ix_(idx['y'],b)]
    Cyb=C[np.ix_(idx['y'],b)]; Cby=C[np.ix_(b,idx['y'])]
    print("eps=%.2f |H_yx|=%.2e |H_yx-pred|=%.2e |Q_ya-C_ya/2|=%.2e |drift+(Cyb+Cby')/2|=%.2e |marg-(Cyb-Cby')/2|=%.2e"%(
        eps,np.abs(Bk(H,'y','x')).max(),np.abs(Bk(H,'y','x')-pred).max(),np.abs(Bk(Q,'y','a')-Bk(C,'y','a')/2).max(),
        np.abs(drift+0.5*(Cyb+Cby.T)).max(),np.abs(marg-0.5*(Cyb-Cby.T)).max()))
