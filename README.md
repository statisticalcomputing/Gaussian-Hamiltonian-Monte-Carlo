# Gaussian-Hamiltonian-Monte-Carlo

## Basic Idea

Like riemann HMC, but not same.

It just transform spatial coordinate and simulate.

It is simple and effective, and is more accurate in some cases.

## Files

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
