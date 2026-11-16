# Simulation summary

- Customers: **100,000** (treatment 49,957 / control 50,043)
- Payments in the 4-week test: **1,434,435**
- Scam payments: **917** (0.064% of payments), £2,145,702 attempted
- Scam mix: purchase 431, impersonation 289, investment 197
- Existing risk score ROC-AUC: **0.926**; flag threshold 0.224 flags 5.0% of payments and **67% of scams**
- Warnings shown (treatment only): 35,917

## Arm balance on pre-period behaviour (should be near-identical)

| Arm | Pre payments | Pre new payees | Pre spend (£) | Tenure (months) |
|---|---|---|---|---|
| control | 28.68 | 5.73 | 3,078 | 60.5 |
| treatment | 28.75 | 5.73 | 3,078 | 60.6 |

## Ground truth (hidden from the analysis)

- Warning everyone would cut scam losses by **13.1%** (£2.80 per customer over the test)
- Cost: 3.1% of warned genuine payments abandoned; 0.0146 extra support contacts per customer
- Share of flagged scam £ stopped, by type: impersonation 35%, investment 7%, purchase 24%
