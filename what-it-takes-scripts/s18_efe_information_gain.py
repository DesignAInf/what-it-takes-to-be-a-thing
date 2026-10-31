"""(i) F 8.14 (Friston 2019, p. 89): the stated condition q(eta|pi)=p(eta|alpha) does not give the information-gain form.
    Binary example: p(eta)=1/2, p(s=eta|eta)=0.9, q(eta,s)=1/4 (alpha fixed).
    F 8.8 under the stated condition, the right-hand side of F 8.14, the corrected form E[I(pi)] + E_q(s) D[p(eta)||p(eta|s)],
    and the information-gain form under the alternative predictive structure q(eta,s)=p(eta|s)p(s), q(eta)=p(eta).
(iii) Two functionals: G (Friston 2019, F 8.8: entropy of q(eta|s,alpha)) and G_IG (entropy of q(eta|alpha), the
    information-gain functional of later work). For one predictive joint density, G - G_IG = I_q(eta;S|alpha) >= 0.
    At convergence q=p: G = I(alpha)+H(S|alpha), G_IG = I(alpha)+H(S|eta,alpha); here I(alpha)=0.
(ii) F 8.11: at convergence G(alpha) = I(alpha) + H(S|alpha); for a continuous (Gaussian) sensory state with sd 0.1
    the differential entropy is negative, so G(alpha) < I(alpha); for discrete states H(S|alpha) >= 0."""
import numpy as np
L=np.log
pe=np.array([.5,.5]); ps_e=np.array([[.9,.1],[.1,.9]])          # rows eta, cols s
pj=pe[:,None]*ps_e; p_s=pj.sum(0); pe_s=pj/p_s[None,:]          # p(eta|s)
q=np.full((2,2),.25); q_e_s=q/q.sum(0)[None,:]; q_s=q.sum(0)
G88=(q*(-L(pj))).sum()-(q_s*(-(q_e_s*L(q_e_s)).sum(0))).sum()   # E_q[I(eta,pi)] - H[q(eta|pi)]
Ipi=(q_s*(-L(p_s))).sum()
IG_q=(q_s*(pe_s*L(pe_s/pe[:,None])).sum(0)).sum()
rhs814=Ipi-IG_q
corr=Ipi+(q_s*(pe[:,None]*L(pe[:,None]/pe_s)).sum(0)).sum()
qa=pe_s*p_s[None,:]                                             # q(eta,s)=p(eta|s)p(s)
Galt=(qa*(L(pe[:,None])-L(pj))).sum()
IG_p=(p_s*(pe_s*L(pe_s/pe[:,None])).sum(0)).sum()
print("(i)  F8.8 = %.4f ; RHS of F8.14 = %.4f ; corrected E[I]+E D[p(eta)||p(eta|s)] = %.4f"%(G88,rhs814,corr))
print("(i)  alternative predictive structure: G_IG = %.4f ; E[I(pi)] - E D[p(eta|s)||p(eta)] = %.4f"%(Galt,(p_s*(-L(p_s))).sum()-IG_p))
h=0.5*np.log(2*np.pi*np.e*0.1**2)
pd=np.random.default_rng(0).dirichlet(np.ones(4)); Hd=-(pd*L(pd)).sum()
print("(ii) Gaussian sd 0.1: H(S|alpha) = %.4f nats (negative) ; discrete example: H(S|alpha) = %.4f >= 0"%(h,Hd))

def G2019(q): return (q*(-L(pj))).sum()+(q*L(q/q.sum(0)[None,:])).sum()
def GIG(q):   return (q*(-L(pj))).sum()+(q*L(q.sum(1)[:,None])).sum()
def Iq(q):    return (q*L(q/(q.sum(1)[:,None]*q.sum(0)[None,:]))).sum()
Hs_e=-(pj*L(ps_e)).sum()
print("(iii) at q=p: G = %.6f = H(S) ; G_IG = %.6f = H(S|eta) = %.6f ; G - G_IG = %.6f = I(eta;S) = %.6f"%(G2019(pj),GIG(pj),Hs_e,G2019(pj)-GIG(pj),Iq(pj)))
qr=np.random.default_rng(1).dirichlet(np.ones(4)).reshape(2,2)
print("(iii) random q: G - G_IG = %.6f ; I_q(eta;S) = %.6f ; q=1/4: G = G_IG = %.4f"%(G2019(qr)-GIG(qr),Iq(qr),GIG(q)))
