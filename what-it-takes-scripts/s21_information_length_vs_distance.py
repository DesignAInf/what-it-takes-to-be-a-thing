"""Information length is a path length, not a distance (Gaussian family, Fisher metric ds^2 = (dm^2 + 2 ds^2)/s^2).
(i)  out-and-back of the mean by 3 sigma: length 6, Fisher-Rao distance of the density from itself 0;
(ii) relaxation m = 3 sigma exp(-theta t) at fixed width: length 3; Fisher-Rao distance in the full (m, s) family
     d = sqrt(2) arccosh(1 + (dm^2/2 + ds^2)/(2 s1 s2)) is shorter."""
import numpy as np
t=np.linspace(0,1,200001); m=3*np.sin(np.pi*t); ell=np.sum(np.abs(np.diff(m)))
print("(i)  round trip: information length = %.4f ; Fisher-Rao distance (start, end) = %.4f"%(ell,0.0))
t=np.linspace(0,40,400001); m=3*np.exp(-t); ell=np.sum(np.abs(np.diff(m)))
d=np.sqrt(2)*np.arccosh(1+(3**2/2)/2)
print("(ii) relaxation: information length = %.4f ; Fisher-Rao distance in the full Gaussian family = %.4f"%(ell,d))
