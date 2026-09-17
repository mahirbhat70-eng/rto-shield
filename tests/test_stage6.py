"""
test_stage6.py — contract tests for the structural simulator v2
(src/data/generator_v2.py) and the uplift/OPE evaluation helpers
(src/eval/stage6_uplift_ope.py).

Why these matter: every v1 financial number rests on the four INTERVENTION
CONSTANTS in cost_config.yaml being true. Stage 6 replaces assumption with
measurement on a simulated causal world. These tests pin the parts of that
world that the honest conclusions depend on:

  * the DGP is deterministic (same seed => same data => same conclusions),
  * the positivity assumption actually holds (propensities >= 1% everywhere),
  * potential outcomes are consistent (interventions never increase RTO in
    expectation) and heterogeneous effects stay inside their design bounds,
  * delayed labels / point-in-time features never leak future information,
  * the EL formula and the OPE estimators (IPS / SNIPS / DR) are exactly the
    mathematics they claim to be (hand-computed / exact-identity checks),
  * the committed report tells the story its own numbers support (P3 trails
    P1; DR bias band includes a double-digit outlier) — narrative honesty is
    pinned the same way numeric claims are pinned in test_stage5.py.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.generator_v2 import generate, validate as validate_v2
from src.eval.stage6_uplift_ope import (
    ACTIONS,
    CLAIMED,
    MARGIN_PCT,
    RTO_COST,
    el_action,
    argmin_action,
    realized_loss,
    ope_estimates,
)

N = 4000
SEED = 42


@pytest.fixture(scope="module")
def v2():
    df, truth = generate(N, SEED)
    return df, truth


# ─── Generator contract ────────────────────────────────────────────────────

def test_generator_v2_deterministic():
    df1, t1 = generate(1200, 7)
    df2, t2 = generate(1200, 7)
    assert df1.equals(df2)
    assert t1.equals(t2)


def test_generator_v2_contract(v2):
    df, truth = v2
    # the DGP's own invariants (asserts internally, explodes if violated)
    assert validate_v2(df, truth, N)
    assert set(df["assigned_action"].unique()) <= set(ACTIONS)
    assert set(df["observed_outcome"].unique()) <= {"DELIVERED", "RTO", "DROPPED"}
    # observed label is consistent with the observed outcome, never the truth
    assert (df["rto_label"] == (df["observed_outcome"] == "RTO").astype(int)).all()


def test_generator_v2_non_cod_passthrough(v2):
    """Same semantics as the hardened serving layer: interventions are COD
    tools — non-COD payments are deterministic ALLOW passthroughs."""
    df, _ = v2
    non_cod = df["payment_method"] != "COD"
    assert (df.loc[non_cod, "assigned_action"] == "ALLOW").all()
    assert (df.loc[non_cod, "propensity"] == 1.0).all()
    assert (df.loc[non_cod, "cod_charge"] == 0.0).all()
    cod = ~non_cod
    assert (df.loc[cod, "propensity"] >= 0.01).all(), "positivity (<1% floor) violated"


def test_generator_v2_delayed_labels_point_in_time(v2):
    """prior_* features must only count RESOLVED history (label arrival >= 5
    days later), so prior_rto_count <= prior_orders and labels arrive late."""
    df, _ = v2
    assert (df["prior_rto_count"] <= df["prior_orders"]).all()
    assert df["label_arrival_delay_days"].between(5, 28).all()
    # account age is non-negative and integer
    assert (df["account_age_days"] >= 0).all()


def test_generator_v2_seasonality_and_cold_pincodes(v2):
    """Festive volume ramp + RTO spike, and cold pincodes appearing only in
    the late window (the drift the v1 frozen policy never faced)."""
    df, _ = v2
    ts = pd.to_datetime(df["timestamp"])
    month = ts.dt.month
    late_share = month.isin([9, 10, 11]).mean()
    assert late_share > 1 / 3, "festive months must be over-represented in volume"
    cod = df["payment_method"] == "COD"
    early_rate = df.loc[cod & (month <= 8), "rto_label"].mean()
    festive_rate = df.loc[cod & (month >= 10), "rto_label"].mean()
    assert festive_rate > early_rate, "no festive RTO drift"
    early_pins = set(df.loc[ts < "2026-08-15", "pincode"])
    late = df.loc[ts >= "2026-08-15"]
    cold_share = (~late["pincode"].isin(early_pins)).mean()
    assert cold_share > 0.02, "cold-pincode population must exist"


# ─── Potential outcomes ground truth ───────────────────────────────────────

def test_generator_v2_interventions_reduce_rto_in_expectation(v2):
    """The causal world the whole stage-6 conclusion rests on: every
    intervention's potential outcome is (stochastically) dominated by ALLOW."""
    _, truth = v2
    y_allow = truth["y_allow"].mean()
    for col in ["y_rto_verify", "y_rto_deposit", "y_rto_prepaid"]:
        assert truth[col].mean() < y_allow, f"{col} not dominated by ALLOW"


def test_generator_v2_heterogeneous_effect_bounds(v2):
    """True effects are heterogeneous but inside the designed support —
    and their means are FAR from the frozen 0.80/0.40 deposit assumption
    (the honest point of stage 6: P1 runs on a wrong constant)."""
    _, truth = v2
    assert truth["r_verify"].between(0.20, 0.42).all()
    assert truth["drop_verify"].between(0.03, 0.07).all()
    assert truth["r_deposit"].between(0.35, 0.70).all()
    assert truth["drop_deposit"].between(0.25, 0.55).all()
    assert truth["r_prepaid"].between(0.30, 0.40).all()
    assert truth["drop_prepaid"].between(0.55, 0.70).all()
    assert 0 < truth["p0"].min() and truth["p0"].max() < 1
    # heterogeneity must actually exist (not degenerate constants)
    assert truth["r_deposit"].std() > 0.05
    # the frozen assumption is wrong by design: |0.80 - true mean| is material
    assert abs(truth["r_deposit"].mean() - CLAIMED["DEPOSIT"]["r"]) > 0.2


# ─── EL formula & routing (hand-computed) ──────────────────────────────────

def test_el_action_hand_computed():
    p, V = 0.3, 1000.0
    # ALLOW: 0.3*150 - 0.7*(0.2*1000) = 45 - 140 = -95
    assert el_action("ALLOW", p, V, 0.0, 0.0) == pytest.approx(-95.0)
    # VERIFY (r=0.3, d=0.05): 2 + 0.3*0.7*150 - 0.7*0.95*200 = 2 + 31.5 - 133 = -99.5
    assert el_action("VERIFY", p, V, 0.3, 0.05) == pytest.approx(-99.5)
    # DEPOSIT (r=0.8, d=0.4): 0 + 0.3*0.2*150 - 0.7*0.6*200 = 9 - 84 = -75
    assert el_action("DEPOSIT", p, V, 0.8, 0.4) == pytest.approx(-75.0)
    # constants match cost_config.yaml
    assert RTO_COST == 150.0 and MARGIN_PCT == 0.20
    assert CLAIMED["DEPOSIT"] == {"r": 0.80, "d": 0.40}


def test_argmin_action_routes_each_action_when_it_wins():
    p = np.full(4, 0.5)
    V = np.full(4, 100.0)
    # crafted effect arrays so a DIFFERENT action is optimal on each row
    effects = {
        "VERIFY":   (np.array([0.10, 0.90, 0.10, 0.10]), np.array([0.90, 0.00, 0.90, 0.90])),
        "DEPOSIT":  (np.array([0.05, 0.05, 0.80, 0.05]), np.array([0.90, 0.90, 0.40, 0.90])),
        "PREPAID":  (np.array([0.10, 0.10, 0.10, 0.90]), np.array([0.90, 0.90, 0.90, 0.10])),
    }
    best = argmin_action(p, V, effects)
    # row 0: nothing works well -> ALLOW; row 1: VERIFY dominates;
    # row 2: DEPOSIT with the claimed constants; row 3: PREPAID dominates
    assert list(best) == [0, 1, 2, 3]


def test_realized_loss_hand_computed():
    V = np.full(4, 100.0)
    action_idx = np.array([0, 1, 2, 3])
    y_rto = {
        "ALLOW":   np.array([False, False, False, False]),
        "VERIFY":  np.array([False, False, False, False]),
        "DEPOSIT": np.array([False, False, True, False]),   # row 2: RTO
        "PREPAID": np.array([False, False, False, False]),
    }
    dropped = {
        "ALLOW":   np.array([False, False, False, False]),
        "VERIFY":  np.array([False, False, False, False]),
        "DEPOSIT": np.array([False, False, False, False]),
        "PREPAID": np.array([False, False, False, True]),   # row 3: dropped
    }
    loss = realized_loss(action_idx, V, y_rto, dropped)
    # row 0 ALLOW delivered: -0.2*100 = -20
    # row 1 VERIFY delivered, friction 2: 2 - 20 = -18
    # row 2 DEPOSIT RTO: 150
    # row 3 PREPAID dropped: 0
    assert loss == pytest.approx(np.array([-20.0, -18.0, 150.0, 0.0]))


# ─── OPE estimators — exact identities ─────────────────────────────────────

def _toy_logged_world(n=20000, seed=11):
    """Deterministic-loss world, uniform propensities 0.25: the true value of
    any policy is exactly computable, so estimator properties are testable."""
    rng = np.random.default_rng(seed)
    loss_per_action = np.array([5.0, 15.0, 25.0, 35.0])   # deterministic given action
    a_logged = rng.integers(0, 4, size=n)                  # uniform logging policy
    prop = np.full(n, 0.25)
    loss_logged = loss_per_action[a_logged]
    choice = rng.integers(0, 4, size=n)                    # arbitrary deterministic target
    v_true = float(loss_per_action[choice].mean())
    return choice, a_logged, prop, loss_logged, loss_per_action, v_true


def test_ope_dr_exact_with_perfect_qhat():
    """DR with a perfect Q-hat on deterministic losses is EXACTLY the true
    value (zero variance) — the doubly-robust identity."""
    choice, a_logged, prop, loss_logged, lpa, v_true = _toy_logged_world()
    q_at_pi = lpa[choice]
    q_at_logged = lpa[a_logged]
    _, _, v_dr = ope_estimates(choice, a_logged, prop, loss_logged, q_at_pi, q_at_logged)
    assert v_dr == pytest.approx(v_true, abs=1e-12)


def test_ope_ips_unbiased_under_uniform_overlap():
    """IPS is unbiased when propensities are correct: with uniform 0.25
    coverage and 20k samples the estimate concentrates on the truth."""
    choice, a_logged, prop, loss_logged, _, v_true = _toy_logged_world()
    v_ips, v_snips, _ = ope_estimates(choice, a_logged, prop, loss_logged,
                                      np.zeros_like(loss_logged),
                                      np.zeros_like(loss_logged))
    assert abs(v_ips - v_true) < 2.0        # ~3+ sd at n=20k
    assert abs(v_snips - v_true) < 3.0


def test_ope_snips_equals_ips_under_full_overlap():
    """When the logger deterministically matches the target (true propensity
    1.0), IPS and SNIPS both collapse to the exact true value — zero
    variance, no normalization needed. Additionally, SNIPS survives a
    constant propensity MIS-scaling (self-normalization cancels the 1/p
    weights) while IPS pays exactly the factor 1/p — the variance/bias
    tradeoff that motivates SNIPS under thin overlap."""
    choice, _, _, _, lpa, _ = _toy_logged_world(n=500, seed=3)
    loss_logged = lpa[choice]               # logger took the target's action
    prop_true = np.ones(500)
    v_true = float(loss_logged.mean())
    v_ips, v_snips, _ = ope_estimates(choice, choice.copy(), prop_true,
                                      loss_logged, np.zeros_like(loss_logged),
                                      np.zeros_like(loss_logged))
    assert v_ips == pytest.approx(v_true, abs=1e-12)
    assert v_snips == pytest.approx(v_true, abs=1e-12)
    # constant mis-stated propensity: SNIPS exact, IPS off by 1/p
    prop_bad = np.full(500, 0.25)
    v_ips2, v_snips2, _ = ope_estimates(choice, choice.copy(), prop_bad,
                                        loss_logged, np.zeros_like(loss_logged),
                                        np.zeros_like(loss_logged))
    assert v_snips2 == pytest.approx(v_true, rel=1e-9)
    assert v_ips2 == pytest.approx(4.0 * v_true, rel=1e-9)


def test_ope_ips_zero_outside_support():
    """No overlap => IPS contribution is exactly zero for those rows (the
    estimator never mixes in unmatched actions)."""
    choice = np.array([2, 2])
    a_logged = np.array([0, 1])
    prop = np.array([0.25, 0.25])
    loss_logged = np.array([5.0, 15.0])
    v_ips, v_snips, v_dr = ope_estimates(choice, a_logged, prop, loss_logged,
                                         np.array([25.0, 25.0]),
                                         np.array([5.0, 15.0]))
    assert v_ips == 0.0
    assert v_snips == 0.0
    # DR falls back to the Q-hat model term alone
    assert v_dr == pytest.approx(25.0)


# ─── Committed stage-6 artifacts tell the story their numbers support ──────

def test_stage6_summary_json_pins_honest_orderings():
    with open("reports/stage6_summary.json", encoding="utf-8") as f:
        s = json.load(f)
    sav = s["policy_savings"]
    assert set(sav) == {"P1 Constants (v1)", "P2 Global-learned",
                        "P3 Uplift (heterogeneous)", "P4 Oracle (true effects)"}
    # deterministic seed => these orderings are the committed result:
    assert s["pct_of_oracle"]["P4 Oracle (true effects)"] == 100.0
    assert sav["P4 Oracle (true effects)"] > sav["P2 Global-learned"]
    assert sav["P2 Global-learned"] > sav["P1 Constants (v1)"]
    # the honest headline: heterogeneous uplift TRAILED the constants policy
    assert sav["P3 Uplift (heterogeneous)"] < sav["P1 Constants (v1)"]
    # festive drift is real in the committed run
    assert s["drift"]["test"]["cod_rto_rate"] > s["drift"]["train"]["cod_rto_rate"]


def test_stage6_report_narrative_matches_numbers():
    with open("reports/stage6_uplift_ope_results.md", encoding="utf-8") as f:
        md = f.read()
    for anchor in [
        "## 1. Learned vs true effects",
        "## 2. Policy comparison",
        "## 3. Off-policy evaluation",
        "## 4. Festive drift",
        "## 5. Takeaways",
    ]:
        assert anchor in md
    # the report must not claim uplift beat the constants policy (it did not)
    assert "TRAILS P1" in md
    assert "did NOT beat a good constant here" in md
    # and must not claim DR was single-digit everywhere (P1 ran +20.9%)
    assert "not an oracle" in md
    # the estimation-honesty anchor: logged-data validation would confirm 0.80
    assert "call it confirmed" in md
    assert os.path.exists("reports/stage6/policy_savings.png")
