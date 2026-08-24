# ASQ-3 Dataset — 1,523 Assessments

## Dataset Overview
- **Assessments**: 1,523 records
- **Variables**: 55 standardized variables
- **Developmental Domains**: 5 Domains (Communication, Gross Motor, Fine Motor, Problem Solving, Personal-Social)
- **Item-Level Scores**: 30 questions (6 items/domain, scores: 10, 5, 0)
- **Demographics & Correction**: Gender, Chronological Age, Premature Weeks, Corrected Age, Age Interval
- **Interpretations**: Domain Scores, 3-Zone Cutoffs, Gauss Percentiles, Clinical Interpretations
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Variables Schema (55 Columns)
1. `assessment_id`: Unique assessment index (1 - 1523)
2. `child_id`: Anonymized child ID
3. `gender`: Biological gender (Male / Female)
4. `chronological_age_months`: Age since birth
5. `is_premature`: Prematurity flag (1 if born >= 3 weeks early)
6. `premature_weeks`: Gestational weeks born early (0 - 9)
7. `corrected_age_months`: AAP Adjusted age for prematurity
8. `age_interval_months`: Standard ASQ-3 interval (2 to 60 months)
9-14. `q_com_1` to `q_com_6`: Communication item scores (10, 5, 0)
15-20. `q_gm_1` to `q_gm_6`: Gross Motor item scores (10, 5, 0)
21-26. `q_fm_1` to `q_fm_6`: Fine Motor item scores (10, 5, 0)
27-32. `q_ps_1` to `q_ps_6`: Problem Solving item scores (10, 5, 0)
33-38. `q_soc_1` to `q_soc_6`: Personal-Social item scores (10, 5, 0)
39-43. `score_communication`, `score_gross_motor`, `score_fine_motor`, `score_problem_solving`, `score_personal_social`: Domain sums (0 - 60)
44. `total_score`: Overall composite score (0 - 300)
45-49. `zone_communication`, `zone_gross_motor`, `zone_fine_motor`, `zone_problem_solving`, `zone_personal_social`: Cutoff zones (WHITE, GRAY, BLACK)
50-54. `percentile_communication`, `percentile_gross_motor`, `percentile_fine_motor`, `percentile_problem_solving`, `percentile_personal_social`: Gauss distribution percentiles
55. `clinical_interpretation`: Multi-domain clinical classification (TYPICAL, MONITORING_REQUIRED, INTERVENTION_RECOMMENDED, ASD_ANOMALY_RISK)
