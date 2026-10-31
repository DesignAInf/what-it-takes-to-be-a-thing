"""Friston (2019), eqs. 2.3, 2.7, 2.8 and 8.4, checked on random discrete joint distributions.
2.3: surprisal = inaccuracy + complexity; H(P) = H(P|E) + I(E,P).
2.7: H(A) = H(A|E) + I(E,P) - I(E,S|A).   2.8: E[D[p(eta|pi)||p(eta|alpha)]] = I(E,S|A) = I(E,S) - I(E,S,A).
8.4: F = surprisal + D[q||posterior] = inaccuracy + complexity >= surprisal, for an arbitrary q."""
import numpy as np
rng=np.random.default_rng(1)
p=rng.dirichlet(np.ones(72)).reshape(3,4,2,3)            # p(eta, s, a, mu)
H=lambda q: -np.sum(q*np.log(q))
pE=p.sum((1,2,3)); pP=p.sum(0); pS=p.sum((0,2,3)); pA=p.sum((0,1)); pES=p.sum((2,3)); pEA=p.sum(1)
I2=lambda pxy,px,py: H(px)+H(py)-H(pxy)
pESA=p.sum(3)
HcondPE=-(p*np.log(p/pE[:,None,None,None])).sum()   # H(P|E) = E[-ln p(pi|eta)], computed directly
print("2.3  H(P) - [H(P|E)+I(E,P)]          =",H(pP)-(HcondPE+I2(p.reshape(3,-1),pE,pP.ravel())))
e_pi=p/pP[None]
inacc=-(e_pi*np.log(p/pE[:,None,None,None])).sum(0)       # E_{p(eta|pi)}[-ln p(pi|eta)]
compl=(e_pi*np.log(e_pi/pE[:,None,None,None])).sum(0)
print("2.3  max|surprisal-(inacc+compl)|    =",np.abs(-np.log(pP)-(inacc+compl)).max())
pAlpha=p.sum((0,1)); pEAlpha=p.sum(1); pSAlpha=p.sum(0); IES_Alpha=H(pEAlpha)+H(pSAlpha)-H(p)-H(pAlpha)
IEP=I2(p.reshape(3,-1),pE,pP.ravel())
print("2.7  H(A) - [H(A|E)+I(E,P)-I(E,S|A)] =",H(pAlpha)-((H(pEAlpha)-H(pE))+IEP-IES_Alpha))
e_alpha=pEAlpha/pAlpha[None]
EIG=(p*np.log(e_pi/e_alpha[:,None,:,:])).sum()
IES=I2(pES,pE,pS); IESA_inter=H(pE)+H(pS)+H(pAlpha)-H(pES)-H(pEAlpha)-H(pSAlpha)+H(p)   # I(E,S,A) by inclusion-exclusion
print("2.8  E[IG] - I(E,S|A)                =",EIG-IES_Alpha,";  I(E,S)-I(E,S,A) - I(E,S|A) =",IES-IESA_inter-IES_Alpha)
J=rng.dirichlet(np.ones(24)).reshape(4,6); pb=J.sum(0); b=2; post=J[:,b]/pb[b]; q=rng.dirichlet(np.ones(4))
F=np.sum(q*(np.log(q)-np.log(J[:,b])))
inacc=-np.sum(q*np.log(J[:,b]/J.sum(1))); compl=np.sum(q*np.log(q/J.sum(1)))
print("8.4  F - [surprisal+KL] =",F-(-np.log(pb[b])+np.sum(q*np.log(q/post))),"; F-(inacc+compl) =",F-(inacc+compl),"; F >= surprisal:",F>=-np.log(pb[b]))
