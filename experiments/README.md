# Experiments

Each numbered folder is a self-contained experiment with its own `run.py`, results, and logs.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 00 | `00_example` | Template | End-to-end sanity check of the project setup |
| 01 | `01_hubble_generate` | Done | Sample free-form generations from a Hubble checkpoint to see what its outputs look like |
| 02 | `02_wikipedia_mia` | In progress | Supervised MIA (Loss + Min-K%) on perturbed 1B/100B Hubble, dup=0 vs dup={1,4,16} Wikipedia passages, item-level split |
| 03 | `02_yago_extraction` | In progress | Plain (greedy) training-data extraction of YAGO biography UUIDs on perturbed 1B/100B Hubble, verbatim-match rate per duplication level |

## Convention
- Create new numbered folders (`01_xxx/`, `02_xxx/`, ...) for new experiments — don't edit old ones.
- Each folder contains: `run.py`, `results/`, `logs/`, `figures/`, and `README.md` (observations).
- This README should only contain brief descriptions of each experiment. Detailed setup, results, and observations belong in each experiment's own `README.md`.
