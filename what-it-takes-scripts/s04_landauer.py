"""Arithmetic behind the 'quantum of inference' (Friston 2019, p. 119)."""
import math
kB=1.380649e-23
print("3 k_B T at 300 K = %.4g J"%(3*kB*300))
bits=3/math.log(2); print("3 nats = %.4f bits"%bits)
for T in (300,293): print("Landauer bound for %.3f bits at %d K = %.4g J"%(bits,T,bits*kB*T*math.log(2)))
print("temperature implied by 1.19e-20 J: %.1f K"%(1.19e-20/(3*kB)))
