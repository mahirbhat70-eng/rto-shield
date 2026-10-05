<USER_REQUEST>
DO IT STEP BY STEP 
# RTO-Shield: Pre-Production Audit Prompts

Copy each prompt into your AI IDE or agent (Cursor, Claude Code, Copilot, etc.) with the repo open. Run them in order.

## How to use these safely

- Run each audit in a **fresh session**. Ideally use a **different model** from the one that wrote the code.
- Never let the same agent both **edit the tests** and **judge whether the tests pass**.
- Make the agent paste **raw command output**. Re-run the key commands yourself.
- Treat "all tests passed" as a claim until Prompt 2 shows the tests can fail.
- Thresholds in brackets are **my suggested starting points**, not standards. Adjust them with your own judgement.

---

## Prompt 0: Ground rules (paste at the start of every session)

```text
You are an adversarial auditor of the repository in this workspace (RTO-Shield, a COD return-to-origin
decision engine). Your job is to FIND PROBLEMS, not to confirm that things work.

Rules:
1. Do NOT modify any test, threshold, config value, or reported number to make something pass.
   If something fails, report it. Only edit files inside a new folder named audit/ unless I say otherwise.
2. Every claim you make must be backed by raw evidence: the exact command you ran and its unedited output.
   If you did not run it, say "NOT RUN". Do not infer results from reading code or from existing reports.
3. If you are unsure, say "UNSURE" and explain what you would need. Never guess a number.
4. Separate your report into: CONFIRMED PROBLEMS, SUSPECTED PROBLEMS, THINGS CHECKED AND FINE.
5. Do not summarise away warnings or failures. List each one.
6. At the end, state the single most worrying finding and your confidence in it (low/medium/high).
Reply "Understood" and wait for the first audit task.
```

---

## Prompt 1: Test inventory and count reconciliation

```text
Audit the test suite inventory.

Tasks:
1. Run `pytest --collect-only -q` and report the exact number of collected tests, per file.
2. The repo description says "60 tests", the README badge says 225, and my notes say 225 passed.
   Reconcile these. Which is true? Show the command output.
3. List tests that are skipped, xfail, or marked slow. Show reasons. Run `pytest -rs -rx`.
4. Run the full suite with `-W error::RuntimeWarning` and again with default warnings. Report the
   difference. Then group the ~2,000+ warnings by category and file and identify any that point to
   real bugs (NaN/inf handling, dtype casts, deprecated behaviour that changes results, divide by zero).
5. Find tests with no assertions, tests that only check "no exception raised", and tests that assert
   trivial truths (e.g. assert True, assert x == x, assert len(result) > 0 only).
6. Find any test that reads a frozen result file (reports/*.md, *.json) and compares it to a hardcoded
   number instead of recomputing. List them. These only prove the file was not changed.

Output a table: test file | #tests | #weak/trivial | notes. End with the 5 weakest tests and why.
```

---

## Prompt 2: Sabotage the code to see whether tests can fail (mutation testing)

```text
Verify that the test suite actually detects bugs. Work in a copy of the repo (git worktree or a temp
folder) so the original is untouched.

For each mutation below: apply ONE change, run the relevant tests, record whether any test FAILED,
then revert. Report a table: mutation | file/line | tests that failed | verdict (CAUGHT / SURVIVED).

Mutations to apply to src/:
1. Expected-loss formula: flip the sign of the margin term.
2. Expected-loss formula: change the RTO cost from 150 to 15, then to 1500.
3. Margin rate: change 20% to 2%, then to 90%.
4. Action selection: replace argmin with argmax.
5. Thresholds: change VERIFY 0.20 to 0.90 and PREPAID 0.48 to 0.05.
6. Friction/verification cost: set to 0, then to 1000.
7. Calibration: bypass the calibrator (return raw model scores).
8. Features: drop the historical pincode RTO feature (fill with a constant).
9. Split: make the temporal split random (shuffle before splitting).
10. Labels: shuffle y_train randomly before fitting.
11. Serving path: make single-order scoring apply a different feature scaling than batch scoring.
12. Fallback: make the unseen-pincode fallback return 0 instead of the global prior.
13. Good-customer drop-off: set drop probability to 0.
14. Audit log: stop writing the decision fingerprint.

Also run `mutmut` or `cosmic-ray` on src/policy and src/serve if installable, and report the mutation score.

A mutation that SURVIVES means the tests do not protect that behaviour. For each survivor, propose
(do NOT apply) a new test that would catch it. Suggested target: at least 80% of these mutations caught.
```

---

## Prompt 3: Circular and tautological test audit

```text
Review every test file for circular or tautological logic.

For each test, classify it as:
A) INDEPENDENT: the expected value comes from a hand-calculated constant, a closed-form formula written
   inside the test, or an independent library.
B) CIRCULAR: the expected value is produced by calling the same function or the same code path being
   tested (e.g. assert score(x) == score(x), or recomputing the expected loss with the project's own helper).
C) FILE-ONLY: the test only compares against a stored report or JSON.
D) WEAK: it only checks types, shapes, non-emptiness, or ranges that nearly anything satisfies.

Output: counts of A/B/C/D per file, then a list of every B, C and D test with a one-line reason and a
concrete rewrite that would make it independent. Pay special attention to tests of: expected-loss cost
engine, threshold derivation, savings numbers (71,741 / 69,786), Bayes ceiling, Monte Carlo, calibration.
Do not rewrite the tests in the repo; put proposed rewrites in audit/proposed_tests/.
```

---

## Prompt 4: Independent recomputation of the cost engine

```text
Independently verify the expected-loss decision engine without importing the project's policy code.

1. Read configs/cost_config.yaml and the README formula:
   EL(a) = FrictionCost(a) + P_RTO(a) * RTO_cost - P_success(a) * Margin, Margin = order_value * 0.20.
2. Write a NEW minimal implementation in audit/independent_engine.py using only numpy/pandas and the
   config values. Do not import from src/.
3. Pick 15 diverse orders (low/high value, low/high P_RTO, each of the 4 actions expected to win, plus
   edge cases: order value 0, 1, 50, 100000; P_RTO = 0, 0.5, 1). For each, compute by hand (show the
   arithmetic) the EL of all four actions and the chosen action.
4. Run the project's engine on the same 15 orders and compare action choice and EL values.
   Report any mismatch greater than 1e-9.
5. Run both engines on the full held-out test set and compare: chosen action for every order, total EL,
   and savings versus the always-allow baseline. Report exact agreement rate.
6. Explain what P_RTO(a) and P_success(a) mean for each action (how deposits/verification change the
   probabilities) and list every hardcoded assumption used to model that (drop-off rates, reduction factors).
   For each, say where it comes from in the code and whether it is measured or assumed.
7. Verify the threshold claims (VERIFY 0.20, PREPAID 0.48) by solving the closed form yourself and
   state under which order values/margins they hold. Note that they depend on order value, so say whether
   they are really constants.
```

---

## Prompt 5: Data leakage audit

```text
Audit the pipeline for data leakage. Report every feature and how it is computed.

1. Build a table of every model feature: name | definition | when it is known (at order time, at
   dispatch, after delivery) | verdict (SAFE / LEAKY / UNSURE). Flag anything derived from delivery
   attempts, NDR status, final status, cancellation after dispatch, courier outcome, or RTO charges.
2. Historical pincode RTO rate: show the exact code. Is it computed ONLY from orders strictly before each
   row's timestamp? Check for: using the whole dataset, including the row's own label, or using
   validation/test rows. Prove with a test: recompute it for 20 random rows using only earlier rows and
   compare to the pipeline's value.
3. Same check for customer-level history features (prior RTO count, cancellation count, account age).
4. Splits: confirm train < validation < test chronologically with no overlapping timestamps. Check that
   no customer_id, pincode-day, or duplicate order appears across splits in a way that leaks. Report
   overlap counts.
5. Calibration sets: confirm val_cal and val_rep do not overlap, and that the calibrator was never fit on
   val_rep or test.
6. Hyperparameter and threshold tuning: confirm none used test data. Search the git history
   (`git log -p`) for any change made after the test set was first scored (e.g. thresholds or features
   changed after seeing test results). Report each such commit.
7. Run a "too good to be true" check: train a model on each single feature alone and report PR-AUC;
   a single feature with near-perfect score is a leak.
8. Run a label-shuffle check: shuffle y in train, refit, evaluate on test. PR-AUC should be near the base
   rate (about 0.28 on COD). Report the result.
```

---

## Prompt 6: Synthetic data generator audit

```text
The results come from a synthetic data generator in src/data/. Audit it.

1. Read the generator. Write a plain-English description of exactly how RTO labels are produced: which
   variables feed the label probability, the functional form (logistic? thresholds?), noise levels, and
   any interaction terms.
2. Is any model feature a direct or near-direct copy of the label, or generated AFTER the label was drawn
   (so it encodes the label)? Check each feature's correlation with the label and with the latent
   probability.
3. The "Bayes ceiling" PR-AUC 0.3497 only exists because the true probabilities are known. Show the exact
   calculation. Confirm it uses the true generating probabilities on the test set, not fitted ones.
4. Compare models: logistic regression PR-AUC 0.3434 vs LightGBM 0.3433 vs calibrated tree 0.3313.
   Is the logistic form what the generator uses? Explain why a tree adds nothing and what that implies
   about real-world performance.
5. Run the generator with 5 different random seeds and report mean/std of: base RTO rate, PR-AUC of each
   model, savings, % of Bayes ceiling. Are the headline numbers (13.1%, 94.7%) stable or one lucky seed?
6. Check realism: compare generator output to published ranges (COD RTO roughly 20-40%, prepaid
   2-8%). List which realistic phenomena are missing (fraud rings, courier-specific failures, seasonality,
   NDR causes, address quality).
7. Conclude with: which reported results could only hold on this generator, and which might plausibly
   transfer to real data.
```

---

## Prompt 7: Recompute every reported metric from raw predictions

```text
Recompute all headline numbers from scratch. Do not use any stored report values as inputs.

1. Load the frozen model(s) and the held-out test set. Generate predictions.
2. Using only scikit-learn / numpy in audit/recompute.py, compute: PR-AUC, ROC-AUC, Brier score,
   weighted bin MAE for calibration, precision/recall/F1 at the operating point, and the COD-subset
   confusion matrix (TN, FP, FN, TP). Compare with: PR-AUC 0.3313, ROC-AUC 0.6729, Brier 0.1476,
   precision 0.2963, recall 0.8555, TN 1,025 / FP 4,121 / FN 293 / TP 1,735.
3. Recompute the policy results: savings 71,741 (expected) and 69,786 (realized), RTOs prevented 951,
   good-customer drops 820, friction spend 6,488, action mix 18.4/45.2/36.4/0.0.
   Show the formulas used.
4. Recompute derived claims and show the arithmetic:
   - 951 / 2,028 = 46.9% of RTOs eliminated
   - Effective RTO after intervention = (2,028 - 951) / (7,174 - 951 - 820) = 19.9% (confirm the denominator)
   - 13.1% uplift = 71,741 / 547,766
   - 2.0x = 71,741 / 35,919
   - ~Rs 10 per COD order
5. Check claim-matrix.md: for each row, run the listed command, and state whether it truly asserts that
   number or just runs unrelated tests. Flag every row where the command does not specifically verify the value.
6. List every number in README.md and the executive brief that does not appear in the claim matrix.
7. Report all discrepancies with exact values and which file is wrong.
```

---

## Prompt 8: Financial simulation and assumptions audit

```text
Audit the financial policy simulation (src/policy, src/eval/stage5*, stage4*).

1. List EVERY assumption: RTO cost (Rs 150), margin (20%), verification cost (Rs 2), deposit friction,
   per-action drop-off rates, RTO reduction effects of verify/deposit/prepaid, OTP cost. For each: value,
   file/line, and whether it is (a) measured from data, (b) from a published source, or (c) invented.
2. Explain how the simulator decides which good customers drop off and which RTOs are prevented after
   an intervention. Are these effects drawn from the same assumptions the policy was optimised on?
   If yes, say so: that makes the savings partly circular.
3. Build a sensitivity table in audit/sensitivity.csv by varying one assumption at a time:
   RTO cost {75, 150, 300}, margin {10%, 20%, 35%}, verify drop-off {0.5x, 1x, 2x}, deposit drop-off
   {0.5x, 1x, 2x, 3x}, verify effectiveness {0.5x, 1x}. Report savings and whether the multi-action policy
   still beats always-allow and the best single threshold.
4. Find the break-even point: how much higher must the deposit drop-off be before the multi-action policy
   stops beating the baseline?
5. The README policy table shows "Total Loss" getting MORE negative as policies improve. Explain the sign
   convention and propose clearer labels.
6. Explain exactly what the 5,000-draw Monte Carlo samples (data resampling? parameters?). Say whether
   "P(savings > 0) = 100%" reflects real-world uncertainty or only sampling noise on this simulated test set.
7. Report what fraction of legitimate COD orders are lost (820 of about 5,146) and the revenue and
   lifetime-value cost not modelled.
```

---

## Prompt 9: Calibration deep-check

```text
Verify probability calibration properly.

1. On the held-out test set, produce reliability tables (10 equal-count bins) for: overall, COD only,
   each recommended action's population (ALLOW, VERIFY, DEPOSIT), and by pincode tier. Report expected
   calibration error and max bin error for each.
2. Split the test set into 4 chronological chunks and report calibration for each chunk (drift check).
   The notes mention the RTO rate moving from 29.7% to 36.5% in festive peaks: show what happens to
   calibration and to the policy's chosen actions in that period.
3. Check isotonic calibration overfitting: how many distinct isotonic steps exist vs val_cal size; show
   the calibrator's output on a fine grid for monotone but spiky behaviour.
4. Compare uncalibrated LightGBM vs calibrated: Brier, log loss, ECE, savings. The notes say calibration
   costs about Rs 10.8k of expected savings; explain why a better-calibrated model gets lower expected
   savings and whether the 'uncalibrated' savings number is trustworthy.
5. Check that probabilities after intervention (P_RTO(a)) are not used as if they were calibrated when
   they are modelled assumptions.
6. Report any region of the score range where the model is badly miscalibrated and what the policy does there.
```

---

## Prompt 10: Serving path equivalence and robustness

```text
Test the real-time scoring path (src/serve, app.py).

1. Equivalence: take 5,000 random test orders. Score them as a batch and one by one through the serving
   function. Report max absolute difference in probability and the count of differing actions. Expected
   difference: below 1e-9 (floating point) and zero action differences.
2. Robustness inputs, each must either score sensibly or be rejected with a clear error, never crash or
   return NaN:
   - missing fields, null/None/NaN, empty strings, wrong types (string for order value)
   - negative or zero order value, extremely large values (1e9)
   - pincode: unseen, malformed, 5 digits, 7 digits, with spaces, leading zeros, non-Indian
   - future timestamps, very old timestamps, timezone mismatches
   - duplicate order submissions, unicode/emoji in text fields, very long address strings
   - categories not seen in training
3. Fallback behaviour: simulate model load failure, calibrator failure, SHAP failure, and a timeout. What
   does the system return? There must be a safe default (e.g. ALLOW with a logged flag) and an alert hook.
4. Determinism: score the same order 1,000 times; confirm identical output. Check random seeds are not used at inference.
5. Latency: benchmark p50/p95/p99 with and without SHAP, single-threaded, under 50 concurrent requests,
   and after a cold start. The README says p50 about 8 ms and the results say about 3 ms: measure and
   report the true numbers and which configuration each belongs to.
6. Security: the models are pickles. Verify artifact_hashes.json is checked at load time and loading refuses
   a modified file. Confirm no code path loads a pickle from user input or a network path.
7. PII: check that the audit log and error logs do not record names, phones, emails, or full addresses.
```

---

## Prompt 11: Drift and cold-start behaviour

```text
Stress-test behaviour when the world changes.

1. Cold start: build test sets where 0%, 12%, 30%, 60% of pincodes are unseen. Report PR-AUC, calibration
   and savings for each. Confirm the fallback prior is used and its value.
2. Prior shift: resample the test set to RTO base rates of 15%, 20%, 28%, 36%, 45%. Report calibration
   and savings. Note where the policy stops beating the baseline.
3. Covariate shift: shift order values up 30% and down 30%; change the mix of categories; report results.
4. Sudden drop in a major feature (e.g. pincode lookup table missing or stale by 90 days): report degradation.
5. Propose a drift monitoring spec: which statistics, window sizes, and alert thresholds. Implement a
   small script audit/drift_monitor.py that computes them from a stream of scored orders plus later outcomes.
```

---

## Prompt 12: Reproducibility and environment

```text
Verify the project reproduces from scratch.

1. Create a brand-new virtual environment from requirements.txt (not your existing environment). Record
   the Python version. The README says 3.11+, my notes mention 3.13; confirm which versions work.
2. Run the full test suite there. Report pass/fail counts and new failures.
3. Re-run data generation, training, calibration and the test reveal twice with the same seed. Are all
   outputs byte-identical or numerically identical? Compare artifact hashes against artifact_hashes.json.
4. Change the seed. Report variation in key metrics.
5. Check requirements.txt versus requirements-ci.txt: are versions pinned? List unpinned packages.
   Generate a lockfile (uv lock or pip-compile).
6. Check the GitHub Actions workflow in .github/workflows: does CI run the full suite, or only a subset?
   Does it fail the build on test failure?
7. Check for hardcoded absolute paths (like C:\Users\...) and OS-specific code that breaks on Linux.
```

---

## Prompt 13: Documentation versus reality

```text
Cross-check every statement across README.md, DASHBOARD.md, claim-matrix.md, WHAT_BROKE.md, docs/, reports/,
and the dashboard code.

Produce a table of contradictions: statement | file A says | file B says | actual (measured).
Known suspects to check: test count (60 vs 225); latency (p50 ~8 ms vs ~3 ms; p99 15-30 ms vs 15 ms);
dashboard views (4 vs 7); Python version; the claim that PREPAID_ONLY is "proven suboptimal"; "100%
deterministic"; "11/11 pre-registered checks" (where is the pre-registration recorded and dated, and is it
earlier than the test reveal in git history?).
Read WHAT_BROKE.md and list every issue it admits to, and whether each is fixed, with evidence.
Flag every claim that implies real-merchant results when the data is synthetic.
```

---

## Prompt 14: Shadow-mode replay harness (real data)

```text
Build a shadow-mode evaluation harness in audit/shadow/ that I can run on a real merchant's order export.

Input: a CSV with the fields in the Pilot Data Request (order_id hashed, order_date, pincode, order_value,
payment_mode, final_status, category, optional customer history, courier, costs).

Requirements:
1. Validate the schema; report missing/extra columns and null rates; refuse to run if required fields are missing.
2. Chronological split of the merchant's data (earliest 60% train, next 20% calibrate, last 20% test).
   Never random split.
3. Compute point-in-time features only (historical pincode RTO rate and customer history using prior rows
   only). Add a built-in leakage test.
4. Fit/calibrate on the merchant's data, then replay the policy on the test slice.
5. Report: base COD RTO rate, PR-AUC, calibration table, RTOs flagged vs missed, good orders put under
   friction, net savings with the merchant's real RTO cost if provided (else clearly mark the assumed Rs 150),
   and a sensitivity range for drop-off assumptions.
6. Compare to simple baselines on the same data: always allow, block COD in top-risk pincodes, a one-rule
   order-value threshold, logistic regression. The engine must beat the best simple baseline to be worth shipping.
7. Output a one-page report (markdown + CSV) stating assumptions and caveats. Never print customer-level PII.
8. Include unit tests for the loader, the point-in-time features, and the report numbers.
```

---

## Prompt 15: Live rollout guardrails

```text
Design and implement production guardrails around the decision engine (spec first, then code in
src/serve/guardrails.py with tests).

1. Kill switch: a config/env flag that forces ALLOW for all orders instantly, checked on every request.
2. Intervention cap: maximum share of orders that may receive friction per hour/day (e.g. 60%).
   If exceeded, degrade to ALLOW for the remainder and alert.
3. Conversion guard: compare checkout conversion of treated vs control; auto-pause if the drop exceeds a
   configurable threshold with statistical confidence.
4. Holdout control: a random N% of orders always get ALLOW and are logged, so true RTO reduction can be
   measured. Randomisation must be seeded per order id and logged.
5. Timeouts and fallback: a hard latency budget (e.g. 50 ms); on timeout return ALLOW and log the reason.
6. Logging: decision, probabilities, top SHAP features, config version, model hash, timestamp. No PII.
7. Alerts: score-distribution drift, calibration drift (once outcomes arrive), spike in unseen pincodes,
   error rate.
8. Rollout plan: shadow -> 5% slice with control -> 25% -> 100%, with go/no-go criteria at each stage.
Write tests that prove the kill switch, cap, timeout fallback and holdout work.
```

---

## Prompt 16: Privacy and security review

```text
Review the project for security and privacy issues, assuming it will process real Indian customer data.

1. List every place personal data could appear: inputs, logs, audit trail, dashboards, error messages,
   exported CSVs, git history, model artifacts. Flag any phone number, name, email, or full address.
2. Check .gitignore: are data files, models and any secrets excluded? Search git history for secrets or real data.
3. Check pickle loading, file paths, and any use of eval/exec, subprocess or unsafe deserialisation.
4. Check the Streamlit app for input validation and for exposure of internal data.
5. Identify what India's DPDP Act would likely require you to handle (purpose limitation, consent/notice,
   retention and deletion, security safeguards, a data processing agreement with merchants). Mark this as a
   list of questions for a lawyer, not legal advice.
6. Propose data minimisation: which fields could be hashed, bucketed or dropped without hurting accuracy.
```

---

## Prompt 17: Final go / no-go report

```text
Using the evidence from the previous audits (list the audit/ files you rely on), write audit/GO_NO_GO.md.

Structure:
1. One-paragraph verdict: NOT READY / SHADOW-MODE ONLY / READY FOR SMALL LIVE SLICE.
2. Scorecard table: check | result | evidence file | severity of any problem (blocker / major / minor).
   Include: tests can fail (mutation score), independent cost recomputation, leakage, metric recomputation,
   assumptions and sensitivity, calibration, serving equivalence and robustness, reproducibility,
   documentation consistency, privacy.
3. Blockers: things that must be fixed before any real merchant sees it.
4. What the results do and do not prove: explicitly state that all savings figures are from synthetic
   data unless real-data shadow results exist, and quote the shadow-mode numbers if available.
5. Remaining unknowns, each with the experiment that would resolve it.
6. Recommended next 5 actions in priority order.
Suggested minimum bars (adjust to taste): at least 80% of mutations caught; zero leakage findings;
batch vs single-order differences below 1e-9; every claim-matrix number independently reproduced;
on real shadow data the engine beats the best simple baseline net of friction costs with the margin of
improvement larger than the sensitivity range.
Be blunt. Do not soften findings.
```

---

## Quick order of operations

1. Prompts 1 → 3 (can you trust the tests?)
2. Prompts 4 → 7 (can you trust the numbers?)
3. Prompts 5 → 6 (is the data honest?)
4. Prompts 8 → 11 (do assumptions and edge cases hold?)
5. Prompts 12 → 13 (reproducible and consistent?)
6. Prompts 14 → 16 (real data, guardrails, privacy)
7. Prompt 17 (decision)
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-06T00:55:27+05:30.

The user's current state is as follows:
Active Document: c:\Users\vansh\OneDrive\Desktop\rto-shield-prep\rto-shield\claim-matrix.md (LANGUAGE_MARKDOWN)
Cursor is on line: 54
Other open documents:
- c:\Users\vansh\OneDrive\Desktop\rto-shield-prep\rto-shield\claim-matrix.md (LANGUAGE_MARKDOWN)
- c:\Users\vansh\OneDrive\Desktop\rto-shield-prep\rto-shield\dashboard.py (LANGUAGE_PYTHON)
- c:\Users\vansh\OneDrive\Desktop\rto-shield-prep\rto-shield\src\eval\calibration.py (LANGUAGE_PYTHON)
- c:\Users\vansh\OneDrive\Desktop\rto-shield-prep\rto-shield\README.md (LANGUAGE_MARKDOWN)
</ADDITIONAL_METADATA>