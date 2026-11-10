# Data dictionary

`training.csv`: zero-based immutable `row_id`; temperature (°C), time (h), binary target, source Excel rows, label_source, zero_time. The source records label 16 rows experimental_record and 8 rows interpolated_0h. These labels are inherited from the unified analysis rule, not a new raw-data audit.

`outer_splits.csv`: run 1–50, row_id, role; 18 train and 6 validation per run. `inner_folds.csv`: run, fold 1–3, row_id, role. Inner folds preserve outer training row order. `config/runs.json` stores exact index lists, seeds and selected parameters.

`manuscript_test.csv`: six displayed retrospective test conditions with original point IDs. The original eight-condition audit separately records training overlap. Training never reads these test labels.

`results/reference/probability_grid.npz`: probability array indexed [time,temperature], 241×241, from dataset 6 / before.
