# Methodology and assumptions

## 1. Survival model

The two-parameter Weibull survival function is:

`S(t) = exp[-(t / lambda)^k]`

- `t`: module age in years.
- `k`: shape parameter.
- `lambda`: scale parameter.

The hazard function is:

`h(t) = (k / lambda) (t / lambda)^(k-1)`

For k > 1, the modelled hazard tends to increase with age; for k = 1 it is constant; for k < 1 it tends to decrease. This describes the statistical pattern, not the physical cause of retirement.

### Right-censored likelihood

For an observed retirement at age t, the likelihood contribution is the Weibull density f(t). For a module still operating at its last observation age t, the contribution is the survival probability S(t). The MLE maximises the combined likelihood.

Assumptions include independent observations and non-informative right censoring. If modules are clustered by site, manufacturer or cohort, independence may be violated. Consider grouped or hierarchical methods if the data justify them.

## 2. Cohort retirement

Each commissioning-year cohort contains initial module DC capacity in kW and module mass intensity in kg/kW. For a cohort of age a at the start of a year, the expected retirement share during the next year is:

`S(a) - S(a+1)`

Expected retired capacity is initial capacity multiplied by that share.

### Capacity-to-mass conversion

`module waste tonnes = retired capacity (kW) × module mass (kg/kW) / 1000`

Keep module mass separate from inverter, battery, cable and mounting-system waste unless those components are explicitly included.

## 3. Degradation and lifetime are separate processes

Degradation reduces the energy output of a module. It does not automatically mean the module is retired. The demonstration ML helper predicts a degradation target from sample features; it is not a survival model. Use grouped/time-based validation on real module records to avoid leakage between repeated observations from the same asset.

## 4. Material pathways

The function `apply_material_pathways` allocates predicted technical retirement mass into:
1. reuse share;
2. non-reused mass sent to recycling;
3. recovered material from the recycling input;
4. recycling residue and non-recycled residual disposal.

These are user-defined scenario fractions. An arithmetically closed mass balance is not evidence that the pathway is operationally or commercially available.

## 5. Validation and uncertainty

Recommended research extensions:
- compare Weibull with lognormal, gamma and non-parametric Kaplan-Meier estimates;
- stratify by technology or climate where sample sizes allow;
- calculate profile-likelihood or bootstrap confidence intervals;
- test alternative event definitions and competing risks;
- quantify uncertainty in installed base and module mass;
- propagate parameter and scenario uncertainty through Monte Carlo forecast runs;
- validate against independent observed retirements.

A single deterministic forecast should not be presented without sensitivity analysis and assumptions.
