"""Friston (2019), eq. 1.20 (marginal flow lemma), exact linear-Gaussian check.
Flow f = (Q - Gamma) grad I with I = x' P x / 2 (P a random precision), Q constant antisymmetric, Gamma = 0.7 I.
For a conditioning set mu and any set eta (inside, disjoint from or overlapping mu), compare E[f_eta | mu] (from the Gaussian conditional)
with sum_{j in mu} (Q - Gamma)_{eta j} d_j I(mu), where I(mu) = -ln p(mu) is the marginal surprisal."""
import numpy as np
rng=np.random.default_rng(0); n=4
A=rng.standard_normal((n,n)); P=A@A.T+n*np.eye(n); S=np.linalg.inv(P)
B=rng.standard_normal((n,n)); Q=B-B.T; G=0.7*np.eye(n); M=(Q-G)@P          # f = M x
for mu,eta in [([0,1],[0,1]),([0,1],[0]),([0,1,2],[1,2]),([1,3],[3]),([0,1],[2]),([0,1],[1,3])]:
    nu=[i for i in range(n) if i not in mu]
    K=S[np.ix_(nu,mu)]@np.linalg.inv(S[np.ix_(mu,mu)])                   # E[nu | mu] = K mu
    E=np.zeros((n,len(mu))); E[mu,:]=np.eye(len(mu)); E[nu,:]=K           # E[x | mu] = E mu
    lhs=(M@E)[eta,:]
    rhs=(Q-G)[np.ix_(eta,mu)]@np.linalg.inv(S[np.ix_(mu,mu)])
    print("mu =",mu,"eta =",eta,": max |E[f_eta|mu] - lemma| =",float(np.abs(lhs-rhs).max()))
