# RTO SHIELD — 5:00 FINAL PITCH (Recording Master)

**All numbers verified against frozen artifacts (93f1157) AND the live deployed app.**
Setup: app warmed 10 min prior · 1440×900 · bookmarks hidden · cursor enlarged · repo URL ready.

## BEAT 1 · 0:00–0:22 · COLD OPEN — View 01 (hero)
🎙 **SAY (verbatim):**

Every year, Indian e-commerce burns thousands of crores on Cash-on-Delivery returns. The package comes back — the courier cost doesn't. Every team, every vendor, is fighting to predict one coin flip: will this order come back? We stopped predicting the coin flip. We price it. This is RTO Shield, our submission for Track 02: AI Risk Manager.

🖥 **SHOW:**
- View 01 hero: "We don't predict the coin flip. We price it." centered.
- Cursor completely still. No scroll, no hover — let the line land.
- Lower-third: RTO Shield — COD Decision Console

## BEAT 2 · 0:22–1:05 · ECONOMICS + COMPETITOR KILL — View 01 (pan)
🎙 **SAY (verbatim):**

Here's the economics. A returned COD order costs about ₹150 in forward, reverse, and packaging, against a 20% margin — and roughly one in four COD orders comes back. So blocking every "risky" order destroys good revenue, and allowing everything burns cash.
Some teams will pitch you COD intelligence. Great — intelligence is table stakes. A risk score doesn't make a decision. RTO Shield prices four moves for every order — allow it, verify the address for ₹2, take a deposit, or push prepaid — computes the expected loss of each in rupees, and always ships with the cheapest. Not a threshold. An argmin.

🖥 **SHOW:**
- Slow pan down the KPI cards. Hover in THIS order, synced to speech:
- ₹71,741 as you say "costs about ₹150" region → expected-savings card
- ₹69,786 → realized card
- 2.0× → routing uplift card
- 94.7% → Bayes-ceiling card
- On "four moves": rest cursor on the panel "How a decision is priced — four moves, one argmin".
- On "An argmin": one final hover on the hero line. Stop.
- Lower-third: EL = friction + p_rto_after × ₹150 − p_success × (20% × order value)
⚠️ *Only hover numbers you actually name. Never hover something you don't say.*

## BEAT 3 · 1:05–2:45 · LIVE DEMO — View 02 (centerpiece — rehearse most)
🎙 **SAY (verbatim):**

Let me show you — live, not a mock-up.
(click) A ₹852 Home order, Cash on Delivery. Watch the pincode panel — Tier 3, 28.5% historical RTO in that area. Account is 65 days old, third order in 24 hours. I hit Score.
(click, pause 1 beat) 39.4% return probability — calibrated — in about 40 milliseconds, full path, SHAP included. Now the price menu. Allow it: minus ₹44 expected. Verify: minus ₹54.60. Deposit: minus ₹50. Minus means expected profit — and the engine routes to the cheapest: VERIFY, saving ₹10.59 per order versus always-allow. Multiply that by thousands of orders a day.
And it shows its work: TreeSHAP says the COD charge and pincode history push risk up; account age pulls it back. Every decision gets a tamper-evident SHA-256 fingerprint — replayable, auditable.
(click) A random order from the untouched holdout — scored instantly — and there's the ground truth. That's the frozen production artifact, scoring in your browser, right now.

🖥 **SHOW (exact sequence):**
- Click Try it live → View 02 (nav) while saying "Let me show you".
- Click the VERIFY ₹852 preset. Let the form fill; cursor to the "Input intelligence" panel as you narrate pincode/age/orders.
- Click SCORE THIS ORDER → PAUSE one full beat on the verdict. Don't talk over the landing.
- Hover the rows of "Expected-loss price menu · all four moves, one argmin" in spoken order: Allow −₹44 → Verify −₹54.60 → Deposit −₹50. Then point at the ROUTED / argmin VERIFY badge and the ₹10.59 saved line.
- On "shows its work": scroll to "Why this price · TreeSHAP on frozen model" — point at the up-bars (COD charge, pincode) then the down-bar (account age).
- On "SHA-256": scroll to "Audit receipt" — hover the ✓ verified line once.
- On "random order": click 🎲 Random holdout order. Let the new verdict render fully.
- Cursor: circle the verdict block + ROUTED badge (slow, one loop).
- Lower-third: P(RTO) 39.4% · argmin: VERIFY_ADDRESS (−₹54.60) — use the ON-SCREEN millisecond figure if you overlay latency; don't hardcode 39.9 ms if the live run shows different.
🎲 *If the random order's ground truth is a MISS (~27% of draws), say, without breaking pace:*
*"Missed — and that's exactly why we price in distributions, not certainties."*
*Then continue to Beat 4. Do NOT re-roll on camera.*

## BEAT 4 · 2:45–3:30 · NOTHING BEATS THE LINE — View 03
🎙 **SAY (verbatim):**

The obvious question: why not just pick a threshold? So we swept every cutoff on 3,673 calibration-window orders. The best single line you could ever find — a VERIFY cutoff at 0.20 — still loses to per-order pricing by ₹18,326. That's 6% better than the best threshold, ever. And it generalizes: on the untouched test window, argmin routing earns 2.0× the savings of the best single-threshold policy.
High-value, low-risk orders get allowed. Small risky orders get a ₹2 phone call. Big risky orders get a deposit. A single line can't do all three at once. A price menu can — because risk isn't a label, it's a surface.

🖥 **SHOW:**
- Nav to View 03 — Policy Frontier ("Why no threshold").
- Cursor traces the amber curve to its lowest point as you say "best single line… 0.20".
- Move to the argmin routing line on "still loses… ₹18,326".
- Point at the annotation EDGE OF THE LINE +₹18,326 / 6.0% on "6% better".
- On the final sentence ("risk isn't a label, it's a surface"): sweep the cursor slowly ACROSS the whole curve — one pass, low to high value. Then freeze.
- No clicks except nav. Slow hands.

## BEAT 5 · 3:30–4:25 · PROOF IT'S REAL — View 04
🎙 **SAY (verbatim):**

Forecasts are cheap — so we paid for evidence. On 7,174 test COD orders, expected savings: ₹71,741. Realized savings — scoring actual outcomes through the same cost engine: ₹69,786. Within 2.7% of forecast. And in 5,000 Monte Carlo draws, every single draw stayed profitable — the worst 5th percentile is ₹63,935. Profit uplift: 13.1%, inside the pass band we pre-registered before looking: 8 to 18.
Now, precision at the operating point is 0.30 — and that's fine, because a false positive here is a ₹2 phone call, not a lost customer. Recall is 0.86 — we catch 86% of returns. And the model itself? It extracts 94.7% of the Bayes ceiling — the theoretical maximum signal on this problem. No model family does meaningfully better. The value lives in the decision layer. So that's where we built ours.

🖥 **SHOW (hover chain, no clicks):**
- Expected ₹71,741 card → 2. Realized ₹69,786 card (rest on "2.7%") →
- Panel "Monte Carlo · 5,000 draws on intervention effects" — point at the P5 ₹63,935 band →
- Panel "Operating point · VERIFY + DEPOSIT = positive" — rest here for 13.1% / pre-registered [8, 18] →
- Panel "Per-action calibration · the price is honest per shelf" — rest for precision 0.30 / recall 0.86.
- "94.7% of the Bayes ceiling" is said VERBALLY here — don't navigate back to View 01 mid-beat; stay on calibration table.
- If time feels tight: the provenance panel is NOT spoken — skip it, don't scroll it.

## BEAT 6 · 4:25–5:00 · CLOSE + THE ASK — View 01 (hero)
🎙 **SAY (verbatim):**

Under the hood: a calibrated LightGBM, an 18-signal contract where every feature is knowable at order time — no leakage — 60 tests green in CI, median latency 8 milliseconds. And every number I've said today is reproducible from the repo — each claim maps to a test.
Razorpay — this is a routing layer you can drop behind checkout. Order comes in. A rupee-priced decision goes out. An audit trail goes to the ledger.
Everyone else will tell you which orders are risky. We tell you what to do with each one — and what it's worth.
We don't predict the coin flip. We price it.

🖥 **SHOW:**
- Nav back to View 01 hero during the first sentence.
- Steady on the hero through "drop behind checkout" — no hovers.
- Final line: hold the hero 3 full seconds in silence → fade to end card.
- End card: 
  - RTO Shield — Track 02: AI Risk Manager
  - github.com/mahirbhat70-eng/rto-shield
  (large, centered, ≥3s readable).
