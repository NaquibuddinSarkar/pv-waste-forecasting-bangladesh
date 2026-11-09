# Empirical data provenance record

Use this template for every real dataset added to the PV forecasting project. Keep a filled copy next to the dataset or in the research appendix.

| Field | Record |
|---|---|
| Dataset file name |  |
| Dataset title |  |
| Publisher / data owner |  |
| Source URL / DOI |  |
| Publication or release date |  |
| Date downloaded |  |
| Geographic coverage |  |
| Time coverage |  |
| Sampling unit | Module / system / project / cohort / other |
| Variables and units |  |
| Retirement event definition |  |
| Censoring definition and observation end date |  |
| Module technology coverage |  |
| Licence and reuse conditions |  |
| Cleaning steps |  |
| Unit conversions |  |
| Known missingness or bias |  |
| Researcher notes |  |

## Cohort data checks

- Is capacity DC module nameplate in kW, rather than AC inverter rating?
- Is commissioning year defined consistently (the model assumes installation at the start of the calendar year)?
- Are duplicate projects or repowered capacity double-counted?
- Are cancelled or not-yet-commissioned projects excluded?
- Does each cohort have a defensible module mass intensity?

## Lifetime data checks

- Are observed retirements distinguished from modules still operating at last follow-up?
- Is the censoring time measured from commissioning?
- Are early replacements, repowering and partial replacements identified?
- Is retirement independent of observation loss after conditioning on available variables? If not, the censoring assumption may be violated.

Do not publish personal, commercially confidential or otherwise restricted asset records without permission and appropriate anonymisation.
