# Does a scam warning work, and who should see it?

[![CI](https://github.com/datawitholoye/scam-warning-experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/datawitholoye/scam-warning-experiment/actions/workflows/ci.yml)

A bank is losing money to **authorised push payment (APP) scams**, where customers are tricked into sending money themselves. Product wants to show a scam warning before risky payments. But every warning also slows down genuine customers, and some give up on payments they meant to make.

This project answers the questions a product squad would actually ask:
1. **Does the warning reduce scam losses?** A properly designed A/B test.
2. **What does it cost?** Guardrails on genuine payments abandoned and support contacts.
3. **Who should see it?** Uplift modelling to target warnings where they change behaviour.
4. **What should we do?** A one-page decision memo for the PM.

## Why simulated data, and why that's a strength
Banks don't publish experiment data on scam warnings. So the data comes from a simulator in which **the true effect of the warning is set in advance** and hidden from the analysis. That makes it possible to check whether each method (plain A/B readout, CUPED, uplift models) actually *recovers the truth*, something you can never check on real data.

Every assumption lives in [`configs/simulation.yaml`](configs/simulation.yaml), and the design is explained in [`docs/simulation_design.md`](docs/simulation_design.md).

## Progress

| Day | Deliverable | Status |
|---|---|---|
| 1 | Simulator: customers, payments, scams, existing risk score, customer-level randomisation, hidden potential outcomes | ✅ |
| 2 | A/B test analysis: power, CUPED, guardrails, the cost of peeking | ⏳ |
| 3 | Uplift model: who should get the warning | ⏳ |
| 4 | Decision memo, final README | ⏳ |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make simulate     # writes data/ and reports/simulation_summary.md (~15 s)
make test
```

## Project structure

```
configs/simulation.yaml     every assumption about the world
src/scam_experiment/
  assign.py                 hash-based, customer-level randomisation
  simulate.py               customers -> payments -> scams -> risk score -> warning outcomes
docs/simulation_design.md   what is simulated, and why
reports/                    generated summaries
tests/                      randomisation, balance, potential-outcome consistency, reproducibility
```
