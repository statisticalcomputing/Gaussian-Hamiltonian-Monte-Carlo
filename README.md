# Hamiltonian Monte Carlo Based on Coordinate Transformation

## Basic Idea

It transform spatial coordinate and simulate.

It is more accurate.

## Files

ghmc_multi.py:  100 line full function sampler

hmc_vanilly.py: corrected ap

demos.m
call ghmc

ghmc.m:
gaussian hmc

basic-idea.m:
show basic idea, i.e. our approach

cov-9999.m
sample a bivariate normal with cov [1 .9999;.9999 1]

cov-999999.m
sample a bivariate normal with cov [1 .999999;.999999 1]

demos.py
ghmc.py


mvn-test.m:
sample mvn

riemann.m
show riemann HMC's approach

vanilla.m:
show vanilla HMC's approach
