# Alignment with the Supporting Information

Audit date: 2026-11-09. Reference: completed full SI with instrument/main-text alignment and the user-edited Chinese Section 7 from the same date; hashes are recorded in `validation/SI_alignment.json`. The related SI conversation was read locally, including the instruction to distribute GPC code only and retain a GitHub-link placeholder. Conversation transcripts are not distributed.

| Item | Package / SI comparison |
|---|---|
| Primary input | 6.xlsx, 24 unique conditions: match |
| Record origins | 16 experimental records and 8 zero-time interpolated labels: match |
| Experimental count | 15 series conditions plus the 25 °C / 12 h control; shared 120 °C / 10 h counted once |
| TEM labels | Documentation now specifies the SI criterion: obvious Pd particles exceeding 5 nm = class 0; absence = class 1 |
| Outer evaluation | 50 saved splits, 18 train / 6 validation, 3 per class held out: match |
| Interpolation | Interpolated labels can enter holdouts; not experimental-only external validation: match |
| Inner optimization | Fold-specific scaling, stratified three-fold pooled OOF log loss, 60 TPE trials, 15 startup trials, queued unit RBF: match |
| Kernels and ranges | RBF / Matérn 1.5 / Matérn 2.5; variance 0.1-100; standardized length scales 0.05-10: match |
| GPC settings | Logistic likelihood, Laplace approximation, optimizer=None, max_iter_predict=200, random_state=42: match |
| Seeds | Outer 202601-202650; inner 203601-203650: match |
| Ensemble | Equal mean of 50 probabilities, threshold 0.5: match |
| Table 15, before column | All seven metric means and sample SDs match reported four-decimal rounding |
| Table 16 | All six probabilities and labels match reported six-decimal rounding |
| Test interpretation | Retrospective; six conditions excluded from before-stage fitting and optimization: match |
| After-addition results | Outside this package; SI additionally describes 30 total records and retuned models |
| Sensitivity dataset | Dataset 6.1 is outside this package |
| Plot symbols | Main-text marker scheme retained; not a replica of the SI-specific interpolation/test symbols |

The earlier README said a 5 nm criterion had not been established. It has now been replaced with the criterion explicitly stated in the SI; no labels or fitted models were changed. The Chinese source's phrase "16 series experiments" is best read as 16 total experimental records: the English full SI and the training table explicitly identify the room-temperature control.

The package is consistent with the primary **before-addition GPC** description. It must not be described as a complete reproduction of all SI analyses. The SI is undergoing further formatting/wording edits in another conversation; no SI document was modified by this audit.
