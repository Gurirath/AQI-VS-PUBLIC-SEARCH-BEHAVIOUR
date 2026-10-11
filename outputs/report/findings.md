# Findings

Weekly PM2.5 (OpenAQ) against Google Trends search interest for Bangkok, Kraków, Melbourne and Lahore, 2022–2024. This file holds the interpretation; the notebooks hold only code and outputs. Every number below is printed in the notebook named in the section heading, and the tables are saved in `data/processed/`.

## Key finding: health searches follow the season, not the air

Searches about symptoms and treatment (hospital, breathing difficulty, cough medicine, eye drops, chest pain) look related to PM2.5 in places, but only because both peak in the same months.

- Lahore "cough medicine" correlates with PM2.5 at **+0.58** raw and **−0.04** once a seasonal fit is removed from both series. Kraków shows the same collapse: cough medicine +0.56 → +0.16, chest pain +0.43 → −0.04, breathing difficulty +0.25 → −0.04.
- **0 of 26** dense health series are significant after the season is removed and the tests are corrected for multiplicity. In the same four cities, 15 of 22 awareness and protection series are significant: every one outside Melbourne except Lahore's generic "mask".
- Most health series have no raw correlation to lose in the first place. The median raw correlation of health series is 0.02 in Bangkok, 0.06 in Kraków, 0.15 in Melbourne and −0.05 in Lahore. The confound is visible in the few series that share the winter calendar, cough medicine above all.
- Awareness and protection behave differently. Bangkok "dust" goes +0.66 → +0.51, Kraków "smog" +0.77 → +0.63, Lahore "smog" +0.60 → +0.58. Those series lose a little to the season and keep the rest.

The practical reading: a raw correlation between pollution and health-seeking searches should not be read as a health response. Cold-season illness and cold-season pollution arrive together.

## Config choices made on keyword validity

These two choices were made on what the keywords mean, before looking at results.

- **Health headline is "breathing difficulty" in every city** (Thai and Polish versions in Bangkok and Kraków, English in Melbourne and Lahore). It is the symptom most specific to air pollution. "Hospital", the previous headline, is searched for every reason people go to hospital.
- **Lahore's protection headline is "air purifier" at country level.** "Mask" is a generic keyword in Pakistan (face masks, cosmetic masks, the pandemic) and is not specific to pollution. Its correlation with PM2.5 once the season is removed is 0.20 and not significant; "air purifier" is 0.61.

## 01 Data collection

- Raw pulls: 182 candidate PM2.5 sensors within 25 km of Bangkok, 5 for Kraków, 25 for Melbourne, 2 for Lahore.
- Search series on disk: 20 for Bangkok, 19 for Kraków, 12 for Melbourne, 11 for Lahore. Awareness and protection terms are pulled for the city's own region where Google reports one; health terms are pulled at country level, and Lahore's English baseline terms are country level only.
- Both collectors skip files that already exist, so the notebook is safe to re-run and makes no requests when the raw data is complete.

## 02 Cleaning and merge

- A sensor joins the city panel only if it is a reference-grade monitor with valid data on at least half of the study days. Low-cost community sensors join part-way through and would shift the city average for reasons unrelated to air quality.
- Kept: 64 of 182 sensors in Bangkok, 2 of 5 in Kraków, 8 of 25 in Melbourne, **1 of 2 in Lahore**.
- Weekly coverage of the 158-week window: Bangkok 150 weeks (95%), Kraków 134 (85%), Lahore 123 (78%), Melbourne 96 (61%).
- Weeks run Sunday to Saturday and are labelled by their Sunday, on local calendar days. The merged wide table `all_cities_combined.csv` has 503 rows and 11 columns.

## 03 EDA

**Data quality.** All four cities lose the same six weeks in February–March 2023; that hole is in OpenAQ. Melbourne is missing about a year. Later methods work on a regular weekly index so that weeks either side of a gap are never paired at a lag. A search series that is zero in more than half of all weeks is dropped: mask searches in Bangkok and Kraków, most English terms at region level in Kraków, Urdu "smog" in Lahore, "P2 mask" in Melbourne.

**Four pollution regimes.** Median weekly PM2.5 is 22 µg/m³ in Bangkok, 15 in Kraków, 6 in Melbourne and 92 in Lahore. The share of weeks above the WHO 24-hour guideline of 15 is 74% in Bangkok, 50% in Kraków, 1% in Melbourne and 100% in Lahore. The design is a heavily polluted city, a seasonally polluted one, a clean baseline, and an extreme case. PM2.5 is right-skewed everywhere, so rank statistics are reported next to Pearson correlations.

**Seasonality and memory.** Bangkok peaks in the dry season, Kraków in the heating season, Lahore in the winter smog season; search follows the same calendar in all three. The annual cycle is strong in Bangkok and Lahore (PM2.5 autocorrelation at one year 0.42 and 0.47), weaker in Kraków (0.16) and absent in Melbourne. Neighbouring weeks are correlated for about ten weeks, so ordinary p-values overstate the evidence. STL with one fixed annual shape is shown for Bangkok, Kraków and Lahore; Melbourne's year-long gap is too long to interpolate.

**Two variables.** Raw correlations with PM2.5 are 0.5–0.8 for the dense awareness and protection series in Bangkok, Kraków and Lahore, and near zero in Melbourne. Rolling 26-week correlations are usually high in the three polluted cities but drop sharply for stretches. Raw lag profiles peak at lag zero for the awareness and protection headlines in all three; Melbourne has no usable peak.

**Many variables.** The first principal component carries 84% of the variance of Bangkok's eight dense awareness and protection series, 77% of Kraków's four and 77% of Lahore's four, against 50% of Melbourne's six. That component is the per-city **attention index**. Health terms are kept out of it.

## 04 Statistics

Method: a seasonal fit (linear trend plus three annual sine/cosine pairs) is subtracted from both series, and the residuals are correlated. Intervals come from an 8-week block bootstrap, p-values from a circular-shift test, and all 52 tests are adjusted with Benjamini–Hochberg.

| City | Attention index, raw | Season removed (95% interval) | Verdict |
|---|---|---|---|
| Bangkok | 0.70 | 0.53 (0.37 to 0.66) | Survives |
| Kraków | 0.75 | 0.58 (0.34 to 0.71) | Survives |
| Lahore | 0.57 | 0.59 (0.03 to 0.80) | Survives, with a very wide interval |
| Melbourne | −0.06 | 0.07 (−0.15 to 0.19) | No relationship |

- The shift test has a floor of about 0.007 with this many weeks, which becomes 0.020 after adjustment. Every significant series sits at that floor.
- **Term lag comparison.** On de-seasonalised data every significant series peaks at lag zero, apart from Lahore "smog" at +1 week. Health series peak anywhere from −7 to +7 weeks, none of them significant, which is what noise looks like. A maximum at the edge of the ±8-week window is reported as no peak: 9 of the 52 series, including both Bangkok hospital series.
- **Pollution tertiles.** Median attention rises from the least to the most polluted third in Bangkok, Kraków and Lahore (epsilon-squared 0.58, 0.48, 0.30) and not in Melbourne. For the health headline the grouping explains nothing in Bangkok, Melbourne or Lahore. Kraków's health headline does differ across raw pollution tertiles (epsilon-squared 0.08), which is the seasonal confound again: this test is run on raw data.

## 05 Awareness gap

All results use the attention index in standard deviations.

- **Level or surprise.** Bangkok responds to unexpected pollution almost as much as to seasonal pollution (0.67 against 0.75 SD per 10 µg/m³). Kraków responds to both but more to the seasonal part (0.46 against 1.11). Lahore's surprise effect is 0.10 SD per 10 µg/m³ with p = 0.11, and Melbourne's is indistinguishable from zero. Lahore's coefficients are small per 10 µg/m³ because its PM2.5 moves by hundreds.
- **Switch-on level.** A flat-then-rising line is no better than a straight line in Bangkok, Kraków or Melbourne. In Lahore it is clearly better (AIC gain 34), with attention flat until about 245 µg/m³, but the interval on that point runs from 41 to 290, so the position is not pinned down.
- **Episodes.** Six pollution episodes each in Bangkok, Kraków and Lahore, five in Melbourne. Attention peaks in the week of the spike or the week after and falls to half within a week. The bootstrap bands on the size of the response include zero in every city, so the timing is the more reliable part.
- **Lag asymmetry.** On the surprise component, attention is higher after PM2.5 rises than after it falls. The gap is clear of zero in the same week in Bangkok (0.25 SD, 0.05 to 0.52) and Kraków (0.20, 0.06 to 0.29), and one week later in Lahore (0.37, 0.06 to 0.84). Melbourne shows no gap at any horizon.
- **Over a season.** Among high-pollution weeks, attention builds through the season in Bangkok (+0.07 SD per earlier high week, stable whichever season is left out) and wears out in Kraków (−0.05, also stable). Lahore points the same way as Kraków (−0.09) with p = 0.075.
- **Silent weeks.** Share of top-third pollution weeks with below-median attention: Bangkok 2%, Kraków 9%, Lahore 17%, Melbourne 50%. Melbourne's top third starts at 6 µg/m³, so those weeks are silent because the air is not bad enough to notice.

## 06 Causal, ML and verdict

- **Stationarity.** With the season removed every series is stationary, so the seasonal residuals are used for both sides of every Granger test.
- **Granger.** No test is significant after correction, in either direction, for either role, in any city. This does not show that pollution has no effect. The response happens within the week, and a Granger test only looks at earlier weeks.
- **Forecast skill, attention.** Adding PM2.5 to a calendar-only random forest lifts out-of-fold R² from 0.28 to 0.43 in Bangkok. In Kraków the calendar alone reaches 0.42 and PM2.5 adds nothing (0.38). Lahore is weak either way (0.11 to 0.13) and Melbourne is below zero.
- **Forecast skill, health.** The health headline is not predictable in any city: out-of-fold R² is −0.30 in Bangkok, 0.04 in Kraków, −0.19 in Melbourne and −0.15 in Lahore.
- **RF heatmap, every dense series.** Awareness and protection series in Bangkok and Kraków reach R² of 0.2 to 0.4. Most health series are below zero, meaning worse than predicting the average. The gain panel should be read together with the R² panel: a gain of +0.30 for Bangkok "hospital (Thai)" takes R² from −3.3 to −3.0 and means nothing.

| | Bangkok | Kraków | Lahore | Melbourne |
|---|---|---|---|---|
| Attention correlation, season removed | 0.53, survives | 0.58, survives | 0.59, survives | 0.07, none |
| Health headline correlation, season removed | −0.01 | −0.04 | 0.10 | 0.20, not significant |
| Lag | Same week | Same week | Same week, smog +1 | No peak |
| Granger, either direction | No | No | No | No |
| R² gained from PM2.5, attention | +0.14 | −0.04 | +0.02 | +0.02 |
| Over a season | Attention builds | Attention wears out | Leans to wearing out | No pattern |
| Silent polluted weeks | 2% | 9% | 17% | 50% |
| Search as an awareness proxy | Reliable | Real but mostly seasonal | Real, imprecise | Not informative |

`awareness_scorecard.csv` and `verdicts.csv` hold one row per city and role (attention and health), eight rows each, with a `role` column.

## 07 Health signal

- **Raw against de-seasonalised, every health series.** See the key finding. The largest drops are Lahore cough medicine (0.62), Kraków chest pain (0.47) and Kraków cough medicine (0.40). Some series move the other way: Lahore "hospital" goes from −0.29 to +0.11, because hospital searches there are lower in the smog season.
- **Role contrast.** Headline series per role, raw → season removed. Bangkok: awareness +0.66 → +0.51, protection +0.60 → +0.38, health +0.05 → −0.01. Kraków: +0.77 → +0.63, +0.63 → +0.41, +0.25 → −0.04. Lahore: +0.60 → +0.58, +0.56 → +0.61, −0.04 → +0.10. Melbourne: +0.22 → +0.17, −0.07 → +0.17, +0.17 → +0.20, none significant.
- **Behavioural cascade.** There is no ordering from awareness to protection to health. Awareness and protection headlines peak at lag zero in Bangkok and Kraków, with the awareness interval at exactly zero in both. Lahore peaks at +1 for awareness (interval −7 to +4) and 0 for protection (−1 to +2). The health headline has no dependable peak anywhere: Bangkok −5 (interval −6 to +7), Kraków −5 with an interval that runs off both ends of the window, Lahore no peak. An earlier version of this project reported a cascade; it came from raw data and edge peaks read as lags.
- **Year over year.** Health stays between −0.12 and +0.26 in every city and year. Awareness and protection are positive in most years but not stable: Lahore falls to 0.05 and 0.09 in 2023 from above 0.5 in 2022 and 2024, and Kraków's air purifier falls to −0.01 in 2023. Melbourne 2023 has 7 paired weeks and is skipped.
- **Summary matrix.** Across 52 dense city-series, significant correlations appear only in the attention, awareness and protection rows of Bangkok, Kraków and Lahore.

## Limitations

- **Lahore rests on one reference monitor.** One of two candidate sensors passed the filter. A single station stands in for a city of over ten million, and the interval on Lahore's attention correlation runs from 0.03 to 0.80. Lahore's direction of effect is credible; its size is not established.
- **Health terms are country level.** Health searches are measured for Thailand, Poland, Australia and Pakistan against one city's PM2.5. A real local health response would be diluted by the rest of the country, so the null result for health is weaker evidence than the positive result for awareness. It shows no detectable national signal, not that nobody in the city searched.
- **Google Trends values differ slightly between pulls.** Trends reports a sample, scaled to each series' own peak. Re-downloading a series can move individual weeks by a few points, so correlations may shift in the second decimal place between pulls. The raw files are kept and the collectors skip existing files for that reason.
- **Three years of weekly data.** Episode, habituation and year-over-year estimates rest on five or six episodes and three seasons per city.
- **Gaps in the PM2.5 record.** Melbourne is missing about a year, and six weeks in February–March 2023 are missing in all four cities.
- **Weekly resolution.** The response is faster than a week, so the order of events cannot be established.
- **Search is a relative index.** Levels cannot be compared between cities, only patterns.
- **Kraków rests on two monitors**, and its region-level search covers the whole province.
- **Four cities** illustrate a contrast; they do not prove a general rule.
