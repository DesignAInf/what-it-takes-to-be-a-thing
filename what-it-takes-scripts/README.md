# Verification scripts for *What It Takes to Be a Thing*

These scripts reproduce every symbolic, numerical and arithmetic check listed in the Verification Log of
*What It Takes to Be a Thing: The Free Energy Principle Explained, Tested, and Extended* (Luca M. Possati).
Analytic arguments, inspections of the printed text and appeals to standard results are documented in the book.

In the book's Verification Log, the last column of each row ("Script") gives the script that reproduces that check (`s01` to `s22`). Each script prints the quantities quoted in the log.

## Run

    pip install -r requirements.txt
    python run_all.py

`run_all.py` executes `s01`-`s22` and compares their output with `expected_output.txt`
(numbers compared to 6 significant digits). The simulations (`s02`, `s03`, `s06`, `s20`) use fixed seeds; `s02` and `s03` take a few seconds each, and no script takes more than about a quarter of a minute.

Checked with Python 3.11, numpy 2.4, scipy 1.17, sympy 1.14.

## Contents

| Script | Checks |
|---|---|
| s01 | Friston (2019) eq. 1.8, NESS lemma (symbolic) |
| s02 | eq. 1.8, solenoidal flow leaves the steady state unchanged (simulation) |
| s03 | eqs. 6.19-6.20, Jarzynski equality (simulation) |
| s04 | eq. 10.11, Landauer bound (arithmetic) |
| s05 | eqs. 8.1, 8.22, synchronization map with a non-injective blanket-to-internal map |
| s06 | Aguilera et al. (2022), Appendix C (Lyapunov solution and the simulation of the Ch. 15 figure) |
| s07 | Schroedinger vs Bohm potential; Fisher form of the momentum variance |
| s08 | eqs. 1.2-1.4, 1.9, 1.10, path action, Lagrangian, Hamiltonian |
| s09 | eqs. 1.15-1.18, 2.10, Fisher information and information length |
| s10 | eq. 1.20, marginal flow lemma (linear-Gaussian) |
| s11 | eqs. 2.3, 2.7, 2.8, 8.4, entropy and free-energy identities |
| s12 | eqs. 5.3-5.5, wave function of a NESS density |
| s13 | eqs. 7.7/7.15 (sign of Q_bb), 7.13 (Lorentz force) |
| s14 | eq. 8.25, Laplace free energy |
| s15 | Biehl, Pollock & Kanai (2021), Appendices A and B, examples (A4) and (A11) (floating point, then exact with sympy) |
| s16 | Aguilera et al. (2022), weak-coupling expansions, eqs. 28-33, 47-48 (numerical comparison at three coupling strengths) |
| s17 | expected free energy = risk + ambiguity; Fields et al. (2022) time scales |
| s18 | eqs. 8.7, 8.11, 8.14: the bound on future surprisal, the information-gain form, and the two functionals G and G_IG |
| s19 | eqs. 1.4, 1.11: Euler-Lagrange equation of the action; fixed-endpoint counterexample |
| s20 | eq. 6.17: housekeeping heat at a NESS with solenoidal flow (analytic and simulation) |
| s21 | eqs. 1.17-1.19: information length (path length) versus Fisher-Rao distance |
| s22 | eq. 10.7: the integral terms of the action versus the endpoint term; Freidlin-Wentzell quasi-potential |


## Citation

If you use these scripts, please cite the book (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
