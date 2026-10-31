"""Existence of the synchronization map in a Gaussian model with a Markov blanket.
Da Costa et al. (2021, Lemma 2.1): sigma exists iff ker Sigma_mu,b is contained in ker Sigma_eta,b.
Example: blanket dimension 2, one internal state, so b -> mu(b) is NOT injective, yet sigma exists
when the external mode depends on b through the same linear combination."""
import numpy as np
Sb=np.array([[1.0,0.3],[0.3,1.0]])
Smub=np.array([[0.4,0.2]])            # Sigma_{mu b}
Setab=1.5*Smub                          # Sigma_{eta b}: same kernel
Mmu=Smub@np.linalg.inv(Sb); Meta=Setab@np.linalg.inv(Sb)
kern=np.linalg.svd(Smub)[2][-1]       # basis of ker Sigma_mu,b
print("mu(b) injective?", np.linalg.matrix_rank(Mmu)==2)
print("Sigma_eta,b applied to ker Sigma_mu,b:", np.round(Setab@kern,12))
b1=np.array([1.0,0.0]); b2=b1+Sb@kern   # differs from b1 along ker of b -> mu(b)
print("mu(b1),mu(b2):",np.round(Mmu@b1,6),np.round(Mmu@b2,6)," eta(b1),eta(b2):",np.round(Meta@b1,6),np.round(Meta@b2,6))

# full joint covariance over (eta, b1, b2, mu) with a Markov blanket: Sigma_eta,mu = Sigma_eta,b Sb^-1 Sigma_b,mu
Setamu=Setab@np.linalg.inv(Sb)@Smub.T
C=np.block([[np.array([[2.0]]),Setab,Setamu],[Setab.T,Sb,Smub.T],[Setamu.T,Smub,np.array([[1.0]])]])
print("joint covariance positive definite:",bool(np.all(np.linalg.eigvalsh(C)>0)))
Pr=np.linalg.inv(C); print("precision entry (eta,mu) [0 => eta independent of mu given b]: %.2e"%Pr[0,3])
