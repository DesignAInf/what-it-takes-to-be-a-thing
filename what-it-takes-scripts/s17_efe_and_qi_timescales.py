"""(i) Friston et al. (2023, path integrals): expected free energy = risk + ambiguity, on a random discrete model
    p(s, eta, alpha) = p(s) p(eta|s) p(alpha|s); three expressions of the same quantity (about 1.135 nats).
(ii) Fields et al. (2022): the time scales hbar/(ln2 kB T) and pi hbar/(2 ln2 kB T) at T = 310 K
    (quoted as about 30 fs and 50 fs; computed 35.5 fs and 55.8 fs)."""
import numpy as np
rng=np.random.default_rng(3)
ps=rng.dirichlet(np.ones(4)); pe_s=rng.dirichlet(np.ones(5),size=4); pa_s=rng.dirichlet(np.ones(3),size=4)
P=np.einsum('s,se,sa->sea',ps,pe_s,pa_s); a=1; Pa=P[:,:,a]/P[:,:,a].sum()
p_s=P.sum((1,2)); p_es=P.sum(2); p_e_a=Pa.sum(0); p_s_a=Pa.sum(1); L=lambda x:-np.log(x)
E1=(Pa*(L(p_es)-L(p_e_a)[None,:])).sum()
cost=(p_s_a*L(p_s)).sum(); p_e_sa=Pa/p_s_a[:,None]; IG=(p_s_a*(p_e_sa*np.log(p_e_sa/p_e_a[None,:])).sum(1)).sum()
risk=(p_s_a*np.log(p_s_a/p_s)).sum(); amb=(Pa*L(Pa/p_e_a[None,:])).sum()
print("(i)  E[L(eta,s)-L(eta|alpha)] =",E1,"; cost - info gain =",cost-IG,"; risk + ambiguity =",risk+amb)
hbar=1.054571817e-34; kB=1.380649e-23; T=310
print("(ii) hbar/(ln2 kB T) = %.1f fs; pi hbar/(2 ln2 kB T) = %.1f fs"%(hbar/(np.log(2)*kB*T)*1e15,np.pi*hbar/(2*np.log(2)*kB*T)*1e15))
