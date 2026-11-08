# Methodology

## Lifetime and retirement

The framework estimates a two-parameter Weibull lifetime distribution. The likelihood includes both observed retirement ages and right-censored observations (assets known to have survived at least until the censoring time). The shape parameter controls hazard behaviour; the scale parameter is a characteristic lifetime.

The cohort forecast computes expected annual retirements from installed capacity by commissioning year. Technical retirement is not identical to physical waste generation: early replacements, refurbishment, reuse, export, storage and formal/informal collection need separate flow assumptions.

## Degradation

The optional degradation model is a supervised demonstration using asset age, temperature and irradiance features. Replace synthetic measurements with documented module performance records and validate with temporal or site-level splits. Do not treat feature importance as causal evidence.

## Validation

Use right-censoring definitions consistently, perform parameter uncertainty analysis (bootstrap or likelihood-based intervals), benchmark against alternative lifetime assumptions, and compare forecasted quantities with independent observations where available. Document all unit conversions and sources.

**Data caveat:** bundled CSVs and configuration values are synthetic or illustrative; no result is a Bangladesh national estimate.
