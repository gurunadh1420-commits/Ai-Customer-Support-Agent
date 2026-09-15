# Golden Set annotation guide (AppleSupport)

This guide is for **manual labeling** of `data/processed/golden_set_candidates.csv`.

## Critical rules

1. **`proposed_intent` is not ground truth.**  
   Values look like `MACHINE_CANDIDATE: battery_drain (...)`.  
   They are heuristic suggestions to speed annotation. You may disagree.

2. **Fill `annotator_label` with exactly one intent** from the taxonomy below.

3. **Use `annotator_notes` for edge cases** (e.g. `multi-issue; primary=battery`, `borderline update vs app`).

4. Do **not** change `customer_text`, `agent_text`, or tweet IDs.

5. Label from the **customer message** (primary). Use `agent_text` only as weak context if the customer message is unclear.

---

## Taxonomy (13 intents)

| Intent | Meaning |
|---|---|
| `battery_drain` | Battery loses charge too quickly in normal use |
| `charging_issue` | Charging / charger / cable / port / won’t charge |
| `software_update_issue` | Breakage framed as caused by iOS/macOS/update |
| `hardware_issue` | Physical/component fault (screen, Touch Bar, won’t power on, etc.) |
| `network_connectivity` | Wi‑Fi, cellular data, Bluetooth, AirDrop, **SIM/carrier** |
| `account_access` | Apple ID / iCloud lockout, 2FA codes, password forgot/reset, sign-in |
| `app_store_issue` | App Store storefront/download/balance/update-list problems |
| `app_issue` | A specific app misbehaves (Safari, Messages, Music, Photos, …) |
| `payment_or_refund` | Charges, payment method, invoice, refund, “money back” |
| `subscription_management` | Manage / cancel / renew subscription or Family Sharing plan |
| `storage_issue` | Device or iCloud storage full / not enough space |
| `howto_or_feature` | How-to / “is it possible” / feature question (not a breakage) |
| `other_or_unclear` | Vague, venting, thanks-only, or cannot tell |

---

## Single-label policy

Label the **PRIMARY support need**.

| Situation | Label |
|---|---|
| Battery is the problem; update is only the cause/context | `battery_drain` |
| Several things broke because of an update | `software_update_issue` |
| SIM / carrier / “no SIM” | `network_connectivity` |
| Lock / 2FA / password reset / can’t sign in | `account_access` |
| App Store won’t load / download / balance | `app_store_issue` |
| Messages/Safari/Music/etc. bug | `app_issue` |
| Refund / bad charge / payment method | `payment_or_refund` |
| Manage/cancel/renew subscription | `subscription_management` |
| Too vague / gratitude only / venting | `other_or_unclear` |

### Hard boundaries

- **Battery vs charging:** drain while using → `battery_drain`; plug/charger/cable → `charging_issue`.
- **Update vs app:** many apps/system after update → `software_update_issue`; one app, no update framing → `app_issue`.
- **Update vs hardware:** software regression framing → `software_update_issue`; physical component → `hardware_issue`.
- **App Store vs app:** storefront vs in-app behavior.
- **Payment vs subscription:** money/charge/refund vs manage/cancel plan.
- **Vague vs actionable:** if you would need to ask “what’s wrong?” first → `other_or_unclear`.

### Multi-problem messages

Pick one primary intent using the table above.  
In `annotator_notes`, write `multi-issue` and list secondary themes.

---

## CSV columns

| Column | Who fills it |
|---|---|
| `example_id` | Pre-filled (`GS-0001` …) |
| `customer_tweet_id` / `agent_tweet_id` | Pre-filled (traceability) |
| `customer_text` / `agent_text` | Pre-filled — do not edit |
| `proposed_intent` | Pre-filled machine hint — ignore if wrong |
| `annotator_label` | **You** — required |
| `annotator_notes` | **You** — optional but encouraged on hard cases |

Valid `annotator_label` values are exactly the 13 intent names (snake_case).

---

## Suggested workflow

1. Read `customer_text`.
2. Glance at `proposed_intent` (optional).
3. Choose one `annotator_label`.
4. If unsure between two intents, pick the closer one and note the alternative in `annotator_notes`.
5. Skip inventing new intent names.

When all rows have `annotator_label` filled, this file becomes the Golden Set for later evaluation (not yet).
