# DATA DICTIONARY (Template)

- `fraud` (target): 0/1
- `month` (str) → `month_num` (int): JAN…DEC → 1..12
- `claim_day_of_week` (str) → `claim_day_of_week_num` (int): Monday..Sunday → 1..7
- `annual_income` (float), `claim_est_payout` (float) → `payout_income_ratio` (float)
- `driver_age` (int), `vehicle_age` (int) → ratio/diff
- `liab_prct` (float) → `liab_payout` (float)
