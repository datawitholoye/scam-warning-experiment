# Simulation design

## The world being simulated

```
Customers ──▶ Payments (4-week test) ──▶ Existing risk score ──▶ Flagged? ──▶ Arm?
  age band,      genuine or scam;             (the bank's fraud           │       control: no warning
  pre-period     scam type depends            model, AUC ~0.93)           │       treatment: warning shown
  behaviour      on age                                                   ▼
                                                              Customer heeds or ignores it
```

| Piece | How it works | Why |
|---|---|---|
| **Customers** | Age band, tenure, payment frequency, share of payments to new payees, 8 weeks of pre-test history | Pre-period history is what CUPED uses on Day 2 to reduce noise |
| **Hidden traits** | *Digital confidence* and *vulnerability* drive who gets scammed and who heeds a warning. The bank never sees them | Real customers differ in ways the bank can't observe; the analysis has to work without them |
| **Scams** | Rare (~0.06% of payments), almost always to a new payee, more likely for vulnerable customers. Type depends on age: purchase scams skew young, impersonation and investment skew older | Mirrors the broad pattern in UK APP scam reporting |
| **Scam amounts** | Purchase ~£250, impersonation ~£1,800, investment ~£4,500 (median) | Losses are heavy-tailed, which makes £ metrics noisy. That becomes a central Day 2 problem |
| **Existing risk score** | A noisy score where ~5% of payments are flagged and ~2/3 of scams are caught | The warning can only reach flagged payments. A perfect score would make the test trivial |
| **Randomisation** | Hash of `salt + customer_id`, by customer, 50/50 | One customer is always in one arm, matching how production experiment platforms work |
| **Warning effect (truth)** | Stops 40% of impersonation, 25% of purchase, 10% of investment scams, more for confident customers. 3% of warned genuine payments are abandoned. Some warned customers contact support | Heterogeneous by design, so Day 3's uplift model has something real to find |

## Potential outcomes
For every flagged payment, both outcomes are drawn: *what happens if warned* and *what happens if not*. Only one is observed, depending on the arm. The other is stored in `data/_truth/` and is never read by the analysis code. That gives each method an answer key:

- `ground_truth.json`: the true average effect on scam £ loss, genuine abandonment and support contacts, by scam type and age band.

## Deliberate simplifications
- **Scam rate is higher than UK reality**, so a 4-week test on 100k customers has enough events to analyse. The rate is a single config value.
- Labels (scam or not) are treated as known by the end of the test. In reality reports arrive weeks later, and the rollout discussion on Day 4 covers that.
- Scammers don't adapt to the warning during the test.
- A genuine payer who abandons is counted as abandoned, even though some would retry later.
