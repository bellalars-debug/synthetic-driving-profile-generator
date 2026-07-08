# Methodology

TODO: document, as they are decided:

- NHTS 2022 variables used and why
- Cleaning/filtering assumptions (e.g. which trip purposes, which
  traveler segments count as "employees")
- Feature definitions used for clustering
- Clustering algorithm and choice of number of archetypes
- Synthetic employee profile sampling approach
- Synthetic driving activity profile generation approach
- Validation approach (how generated profiles are checked against NHTS
  distributions)

## Deferred scope

Vehicle type and EV penetration scenarios (5%, 10%, 20%) are
intentionally **not** modeled in this phase. The core generator models
general daily travel behavior only; EV/charging-demand logic will be
added later in `src/driving_profiles/scenarios/`.
