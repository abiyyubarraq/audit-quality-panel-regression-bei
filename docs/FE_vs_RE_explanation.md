# Fixed Effects vs Random Effects: Simple Visual Guide

## The Coffee Shop Analogy

### Scenario: Does WiFi increase coffee sales?

#### Data:
- 3 Starbucks stores (A, B, C)
- 4 years of data
- Some years have WiFi, some don't

---

## Random Effects (RE) - "All stores are similar"

```
Store A (Downtown):     Sales = $10,000/day
Store B (Suburb):       Sales = $3,000/day
Store C (Airport):      Sales = $15,000/day

With WiFi:    Average = $12,000
Without WiFi: Average = $8,000

RE Conclusion: WiFi increases sales by $4,000
```

**Problem**: Maybe downtown/airport stores just sell more anyway!

---

## Fixed Effects (FE) - "Each store is unique"

```
Store A (Downtown):
  2021 (No WiFi):  $9,000/day
  2022 (WiFi):     $11,000/day     ← Up $2,000

Store B (Suburb):
  2021 (No WiFi):  $2,500/day
  2022 (WiFi):     $3,500/day      ← Up $1,000

Store C (Airport):
  2021 (No WiFi):  $14,000/day
  2022 (WiFi):     $16,000/day     ← Up $2,000

FE Conclusion: WiFi increases sales by $1,667 (average within-store change)
```

**Solution**: We compare each store to ITSELF, removing location effects!

---

## Your Audit Data Example

### Company XYZ over time:

```
FIXED EFFECTS COMPARISON:
========================

2021: ARL = 80 days,  AQMS = 2  }
2022: ARL = 90 days,  AQMS = 2  } → "When ARL increases by 10 days
2023: ARL = 85 days,  AQMS = 2  }    within Company XYZ, does AQMS change?"
2024: ARL = 100 days, AQMS = 3  }

This REMOVES the effect of:
✓ Company XYZ being a bank (permanent)
✓ Company XYZ being large (permanent)
✓ Company XYZ's risk profile (permanent)
```

---

## Hausman Test Decision Tree

```
START: I have panel data
  |
  v
Run Hausman Test
  |
  v
Are FE and RE results very different?
  |
  +-- NO (p > 0.05) --> Use Random Effects (more efficient)
  |
  +-- YES (p < 0.05) --> Use Fixed Effects (more consistent)
                         ^
                         |
                    YOUR DATA IS HERE!
                    (p < 0.001)
```

---

## What Hausman Test Actually Tests

**Null Hypothesis (H0)**:
> "Company differences are RANDOM (not correlated with variables)"

**Your Result**:
> Reject H0 (p < 0.001)

**Translation**:
> "Company differences are NOT random - they're systematically related to audit lag, fees, etc. Must use Fixed Effects!"

---

## Key Takeaway

**Random Effects** = ❌ Wrong for your data
- Assumes companies are random samples
- Your companies have permanent differences

**Fixed Effects** = ✅ Right for your data
- Accounts for permanent company differences
- Compares each company to itself over time
- More conservative (harder to find significant results, but more trustworthy)

---

## Your Thesis Should Say:

> "We employ Fixed Effects regression because the Hausman specification test strongly rejected the Random Effects model (χ² = 39.65, p < 0.001). This indicates that unobserved company-specific characteristics are correlated with our independent variables. The Fixed Effects approach controls for all time-invariant company heterogeneity, ensuring our estimates reflect within-company variation over time rather than spurious between-company differences."

**Translation**:
> "We use Fixed Effects because companies are too different from each other. This way, we compare each company to itself, which is more accurate."
