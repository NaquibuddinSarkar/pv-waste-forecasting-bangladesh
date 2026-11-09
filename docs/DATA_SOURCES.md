# Bangladesh PV data sources and research gaps

This register identifies candidate public sources. A source being listed here does not mean that it provides the exact panel-level or historical cohort data needed by the model. Confirm its coverage, definition, publication date, extraction date and reuse terms before use.

## Candidate public sources

| Source | Potential use | Important limitation |
|---|---|---|
| [SREDA National Database of Renewable Energy](https://ndre.sreda.gov.bd/) | Renewable project inventory and capacity context | Verify historical completeness, unit definitions, project status and whether commissioning year is available |
| [SREDA capacity snapshot on the national SDG portal](https://sdg.gov.bd/uploads/indicator-data/Attachment-2%20%28NDRE%29.pdf.pdf) | Point-in-time installed-capacity reference | A snapshot is not a complete historical installation series or a retirement register |
| [Power Division policy portal](https://powerdivision.gov.bd/pages/static-pages/694032be35ce18e1c0561a07) | Current renewable-energy policy and programme context | Policy targets are not observed installation or retirement data |
| [NLR PV Lifetime Project](https://www.nlr.gov/pv/lifetime) | Degradation and lifetime research methods | International test-centre evidence is not automatically representative of Bangladesh's climate, module mix or maintenance practices |
| [Harvard Dataverse: Dhaka PV module dataset (2026)](https://doi.org/10.7910/DVN/Q56G63) | A small local outdoor degradation dataset for method development | Its limited module count, age and sampling design do not establish national fleet performance |

## Data that a defensible Bangladesh forecast still needs

1. Historical PV module capacity by commissioning year, technology and project type.
2. Retirement, replacement and repowering records with the date and a consistent event definition.
3. Follow-up dates for assets still operating, so they can be treated as right-censored rather than failures.
4. Module mass and material composition by model or technology generation.
5. Refurbishment, second-life, export, storage, formal collection and informal recovery pathways.
6. Measured degradation or performance-ratio records with climate and operational context.

The gap between installed-capacity records and module retirement records is a central research limitation. Do not infer a national survival curve from a small field sample or a recent aggregate capacity snapshot.

## Dataset provenance template

For every empirical file, record:
- dataset title and publisher;
- persistent identifier or exact source URL;
- publication/release date and access date;
- geographic and temporal coverage;
- variables, units and definitions;
- licence and reuse conditions;
- cleaning, aggregation and conversion steps;
- missingness, known bias and limitations.

Keep raw source files immutable. Store cleaned datasets separately and retain a transformation script. Never overwrite synthetic demo CSVs with empirical observations without updating their filenames and provenance records.
