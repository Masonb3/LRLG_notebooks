# Project

This project aims to create a readable, efficient code calculating checkerboard games arising in the Lagrangian-Grassmannian. Originally in Python, this will hopefully be done in C later (perhaps even in parallel).

# Authors

The original ideas for this calculation and previously used Maple code were theorized, programmed, and tested by Dr. Leonardo Mihalcea. The python notebooks and subsequent work was done by Mason Beahr. This work was influenced by Ravi Vakil's paper at https://arxiv.org/abs/math/0302294v1

# Overview

When working with Schubert varieties in general, it is common to seek a Littlewood-Richardson Rule to entirely determine the ring structure of the cohomology ring of corresponding Schubert classes. Using (in our case) a Schubert basis, one can write a product as an expansion of terms in the basis, and we wish to understand what coefficients arise and to what term they are attached. In the Lagrangian Grassmannian, there is some particular symmetry that arises and influences these calculations, which we seek to understand. Originally, the algorithm for computing these coefficients was proposed in terms of strict rules dependent on relative positions of so-called 'white checkers' (see Vakil's notation from the authors section). But now, it is theorized that the algorithm can greedily rely simply on a set of geometric invariants. This would allow for (hopefully) much simpler proofs and a stronger understanding of the structure imposed in the Lagrangian-Grassmannian case.