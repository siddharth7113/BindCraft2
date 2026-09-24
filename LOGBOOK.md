# BC2 (BindCraft 2) — Setup & Campaign Logbook

**Machine:** `AIMecoSPECS` · **Repo:** `/home/siddharth/work/BindCraft2` @ `18a9042` (BC2 v1.0.0)
**Project:** stand BC2 up on the local 4-GPU SLURM box; design a 12-mer linear peptide binder against human RBP4.

---

## How this logbook is maintained

*Read this first if you are picking the project up, or adding to it.*

1. **One dated entry per working session**, newest first, under [Entries](#entries). Heading format:
   `### YYYY-MM-DD — Session N: <short title>`.
2. **Each entry has four parts**, in this order:
   - **Goal** — what the session set out to do.
   - **Did** — what was actually run. Include the literal commands, so any step can be repeated.
   - **Found** — results and surprises, *including things that did not work*. Numbers, not impressions.
   - **Open** — what is left, carried into [Open items](#open-items).
3. **Never rewrite a past entry.** If something recorded earlier turns out to be wrong, leave it and
   correct it in the current entry under **Found**, saying which entry it corrects. The record of what
   we believed at the time is the point of the logbook.
4. **Durable facts go in [Reference](#reference), not in entries.** Hardware, paths, commands and
   provenance change rarely; keep one authoritative copy there and update it in place. Entries are the
   narrative; Reference is the current truth.
5. **[Current state](#current-state) is rewritten every session** — it is the "where are we" box, and
   is the only section apart from Reference that gets edited rather than appended.
6. **Record the negative results.** A filter that rejected everything, a card that OOM'd, a job that
   pended — these are the expensive things to rediscover.
7. Cite code as `file.py:line` when a claim rests on it, so it can be re-checked after an upstream pull.

---

## Current state

*Updated 2026-09-24.*

| | Status |
|---|---|
| Install (uv venv, cuda13, jax 0.11.2) | **Working** — `source env.sh` |
| Weights / caches in repo | **Working** — `cache/alphafold` (5.3 GB), `cache/compile`, `cache/protenix_root` |
| Target prepared | **Done** — `targets/RBP4.pdb`, chain A, residues 1–174 |
| Multi-GPU fan-out | **Working** — 3 workers on GPUs 0,1,2, ~4.5 GB each |
| SLURM | **Validated** — `submit_rbp4.slurm`, `--time=0` (no wall limit) |
| **Cyclic campaign** (job 208) | **COMPLETE** — 10/10 designs, 815 trajectories, 10 h 56 m, **1.2 %** accept |
| **Linear campaign** (job 209) | **COMPLETE** — 11/10 designs, 347 trajectories, **3.2 %** accept |
| Protenix 2.0.0 | **Working** — `source env-protenix.sh`, protenix-v2 (464M) verified by SHA256 |
| Orthogonal validation | **COMPLETE, 5 seeds** — 20 designs × seeds 101/202/303/404/505 = 100 design-seeds, 500 structures, 0 errors |
| **Key result** | **r(BC2 i_pTM, Protenix ipTM) = +0.06** over 5 seeds (was +0.08 at 1 seed); r(BC2 i_pTM, jaccard) = **−0.19**. **Do not rank on BC2 i_pTM.** |
| **Best candidate (linear)** | `rbp4lin_r09_l19` `MPEVVKMRSPEGKWEEHTF` — jaccard **1.00 in 3 of 5 seeds**, mean 0.90, ipTM 0.787 ± 0.021. Runner-up `rbp4lin_r06_l24`, ipTM 0.827 ± 0.014 / jaccard 0.80 ± 0.07 |
| **Best candidate (cyclic)** | `rbp4cyc_r08_l14` `TFHRPGQTMVLDGL` — ipTM 0.875 ± 0.036, jaccard 0.72 ± 0.04 |
| ~~Best design (cyclic) = rank 3~~ | **SUPERSEDED (Session 8)** — `rbp4cyc_r03_l17` binds off-epitope in 4 of 5 seeds (jaccard 0.14 ± 0.31). Its BC2 i_pTM of 0.79 was the highest in the set. |
| Consensus shortlist | `analysis/protenix/agreement.csv` — **9 of 20** pass stability gates; per-seed detail in `agreement_per_seed.csv` |
| Ring closure | **Confirmed across seeds** — median N-to-C 1.35 Å over 10 cyclics (1.30–1.41 Å) |
| MD (GROMACS 2026.1) | **Available, not started** — `source /usr/local/gromacs/bin/GMXRC` |

## Entries

### 2026-09-24 — Session 8: five seeds per design; two verdicts were single-draw artifacts

**Goal.** Put error bars on the Protenix numbers before committing to a synthesis shortlist. Session 7
ranked all 20 designs on a single seed (101), and the ranking signal is now Protenix rather than BC2
(r = +0.08), so the shortlist rested entirely on one draw per design.

**Did.**

```bash
python3 analysis/protenix/analyse.py    # FIRST: parity check on seed 101 alone, before new data
./run_protenix_seeds.sh                 # seeds 202,303,404,505 -> 80 new job-seeds, 17m43s
python3 analysis/protenix/analyse.py    # aggregate over 100 design-seeds
```

**Found.**

*A latent bug in `analyse.py` would have inverted this result, and it was caught before the run.*
The old line collected candidate structures with `job.rglob('*.cif')` — **recursive** from the job
directory. With one seed present that saw 5 samples and kept the most confident, which is correct.
With five seeds it would have seen 25 and reported the **maximum over seeds** — a number that only
climbs as seeds are added, whatever the design is worth. That is precisely backwards from an error
bar. Fixed with [`seed_dirs()`](analysis/protenix/analyse.py#L81) and
[`best_sample()`](analysis/protenix/analyse.py#L87): best sample *within* each seed, then mean ± sd
*across* seeds. **Session 7's published numbers are unaffected** — only `seed_101` existed when they
were written. The fix was verified by re-running the new code on seed 101 alone: all 20 rows
reproduced exactly, `px_iptm` and `jaccard` to the digit. Old table kept at
`analysis/protenix/agreement_seed101.csv`.

*The 5 samples per seed are near-duplicates. Seeds carry the variance, samples do not.*

| `rbp4lin_r01_l25` | 5 sample ipTMs | spread |
|---|---|---|
| seed 101 | 0.805 0.806 0.806 0.806 0.809 | **0.004** |
| seed 202 | 0.780 0.783 0.784 0.785 0.785 | **0.005** |
| between the two seeds | | **0.024** |

So `--sample 5` buys almost no diversity — five draws landing on top of each other. **To widen the
estimate, add seeds, not samples.**

*The reseed is minutes, not hours — the earlier "hours" estimate was wrong.* **29 s of GPU per
job-seed** plus a ~100 s one-time model load. 80 job-seeds across GPUs 0 and 1 finished in
**17 m 43 s**, 80/80 succeeded, zero errors. **Zero MSA server calls** — `rbp4*_input.json` carry
absolute `pairedMsaPath`/`unpairedMsaPath` into `analysis/protenix/msa_cache`, so Session 7's cache
work makes reseeding free of the queue that dominated that session.

*Two verdicts flipped, and both were designs where one draw was making the decision.*

| design | seed 101 ipTM/jac | 5-seed mean ipTM/jac | change |
|---|---|---|---|
| `rbp4cyc_r03_l17` | 0.326 / 0.69 | 0.412 / **0.14** (sd 0.31) | **SAME EPITOPE → DIFFERENT SITE** |
| `rbp4cyc_r04_l15` | 0.878 / 0.23 | 0.882 / **0.75** (sd 0.29) | **partial → SAME EPITOPE** |

Seed 101 was the lucky draw for r03 and the unlucky one for r04. On single-seed data r03 would have
been carried onto the shortlist and r04 discarded; **both were wrong.** Neither is a keeper now —
their jaccard sd of 0.31 and 0.29 means the peptide moves around the protein between seeds. The other
18 verdicts held.

*The Session 7 headline strengthens with 5× the data.* r(BC2 i_pTM, Protenix ipTM mean) = **+0.06**
(was +0.08 on one seed). And r(BC2 i_pTM, jaccard mean) = **−0.19** — BC2's own confidence is, if
anything, mildly *anti*-correlated with whether an independent model agrees on the epitope. **Do not
rank on BC2 i_pTM.**

*Shortlist: 9 of 20 survive stability gates.* Gates: mean jaccard ≥ 0.55, jaccard sd < 0.15,
**jaccard min ≥ 0.4 (agreement in every seed, not on average)**, mean ipTM ≥ 0.60, ipTM sd < 0.10.

| # | design | len | ipTM | jaccard | jac min | hotspots | sequence |
|---|---|---|---|---|---|---|---|
| 1 | `rbp4lin_r06_l24` | 24 | 0.827 ± 0.014 | 0.80 ± 0.07 | 0.71 | 5.2 | `EKPKVKLTPGMTMEEFKEEYEKKL` |
| 2 | `rbp4lin_r10_l21` | 21 | 0.730 ± 0.086 | 0.88 ± 0.04 | 0.83 | 5.6 | `LTDEQIVEEFEKHNMKVVHRE` |
| 3 | `rbp4lin_r01_l25` | 25 | 0.810 ± 0.015 | 0.78 ± 0.07 | 0.72 | 6.2 | `APVPKMPEEIKKEAEKWGMKMTYRP` |
| 4 | `rbp4cyc_r08_l14` | 14 | 0.875 ± 0.036 | 0.72 ± 0.04 | 0.67 | 2.0 | `TFHRPGQTMVLDGL` |
| 5 | `rbp4lin_r03_l21` | 21 | 0.822 ± 0.007 | 0.71 ± 0.04 | 0.65 | 5.0 | `SKTKMPEEMKEEMDKHGIKYT` |
| 6 | `rbp4cyc_r07_l16` | 16 | 0.704 ± 0.032 | 0.82 ± 0.09 | 0.73 | 2.6 | `RHPDKTVTLPEMNVTM` |
| 7 | `rbp4lin_r04_l21` | 21 | 0.790 ± 0.008 | 0.71 ± 0.06 | 0.65 | 6.0 | `MIVPAKNDKGEEVPVHMRKRE` |
| 8 | `rbp4lin_r08_l24` | 24 | 0.786 ± 0.031 | 0.60 ± 0.03 | 0.57 | 5.2 | `EWKVWSREEVTEDMKKHGMPVPAP` |
| 9 | `rbp4cyc_r02_l17` | 17 | 0.616 ± 0.045 | 0.69 ± 0.05 | 0.61 | 2.8 | `IEEVNMRMRHPDGTEMV` |

**Two exclusions are knife-edge and should not be read as rejections.** `rbp4cyc_r09_l20` fails on
mean jaccard 0.54 against a 0.55 gate; `rbp4lin_r09_l19` fails on jaccard sd of exactly 0.15 against a
`< 0.15` gate, and it has the **highest mean jaccard in the whole set (0.90)**. Both enter the list on
a hair's different threshold. The gate values are a judgement call, not a standard.

*Cyclics contact far fewer hotspots than linears* — 2.0–2.8 vs 5.0–6.2. They are shorter (14–17 aa vs
21–25), so this is largely length, but it means **the cyclics grip a smaller piece of the portal**.
Worth weighing against their entropic and proteolytic advantages.

*Ring closure still holds across all five seeds* — median N-to-C 1.35 Å over the 10 cyclics
(1.30–1.41 Å), confirming Session 7's 1.27–1.39 Å on one seed. `covalent_bonds` is honoured every time.

*A verification command of mine misreported, and the monitor repeated it.*
`find analysis/protenix/out -maxdepth 2 -name 'seed_*'` returns **nothing** — the seed directories sit
at depth 3 (`out/<set>/<job>/seed_NNN`). A progress monitor built on it announced "0 / 80 job-seeds"
for a run that had in fact completed all 80. Use `-mindepth 3 -maxdepth 3`. Nothing was lost; the
false zero was caught against the per-job count, which was right.

*Correction, same session: my own stability gate is wrong in one direction.* Checking the two
knife-edge exclusions against `agreement_per_seed.csv` settled them, and one exclusion was an artifact:

```
rbp4lin_r09_l19   jaccard per seed:  0.67  1.00  1.00  1.00  0.85
rbp4cyc_r09_l20   jaccard per seed:  0.44  0.60  0.60  0.53  0.53
```

`rbp4lin_r09_l19` hits **perfect agreement (1.00) in three of five seeds** — Protenix reproduces BC2's
residue set exactly. Its sd of 0.15 comes entirely from seed 101 at 0.67, the *lowest* of the five and
the number the old single-seed table reported. **A `jaccard sd` gate penalises variance in both
directions, but upward jaccard variance is not a defect.** The gate should be one-sided, or applied to
`jaccard_min` alone — which this design passes comfortably at 0.67. It belongs in the shortlist and is
the strongest linear in the set on epitope agreement.

`rbp4cyc_r09_l20` is the genuine borderline case: 0.44–0.60 in every seed, stable and unremarkable.
Keep as a backup, not a first pick.

**Revised suggested four:** `rbp4lin_r09_l19`, `rbp4lin_r06_l24`, `rbp4lin_r10_l21`,
`rbp4cyc_r08_l14` — this drops `rbp4lin_r01_l25` (fine at 0.810/0.78, but 25 aa and beaten by r09).

**Open.** The final 3–4 is a judgement call that now needs non-computational input: macrocycle
synthesis feasibility (which bears on the three cyclics) and the Protenix-v2 licence question. The
compute side of candidate selection is done.

---

### 2026-09-23 — Session 7: Protenix validation complete; BC2 ranking does not survive an independent model

**Goal.** Run all 20 designs through Protenix-v2 and test whether an independent architecture agrees
with BC2 on where each peptide binds.

**Did.** `./run_protenix.sh` (see below) -> 20 jobs, 100 structures, 0 errors.
`analysis/protenix/analyse.py` -> `analysis/protenix/agreement.csv`.

**Found.**

*The MSA server was the whole bottleneck, and the fix was not a different server.* ByteDance's
`protenix-server.com` returned HTTP 200 but its queue never advanced (4+ min `PENDING`, no movement).
Switching to `--msa_server_mode colabfold` (`api.colabfold.com`) **did** return a real alignment --
**673 sequences for RBP4** -- but then sat another 10+ min on the *binder* query. The real problem was
**40 redundant searches**: the identical RBP4 target 20 times, plus 20 de novo peptides that by
construction have no homologs to find.

Fix: [`analysis/protenix/build_msa_cache.py`](analysis/protenix/build_msa_cache.py) writes
`non_pairing.a3m`/`pairing.a3m` once for the target (reusing the 673-sequence alignment) and singletons
for each binder. Protenix skips the search when those paths exist
([`runner/msa_search.py:41-56`](bindcraft/../.venv-protenix/lib/python3.12/site-packages/runner/msa_search.py)).

| | before | after |
|---|---|---|
| MSA server calls | 40 | **0** |
| GPU utilisation | 0 % | **93-95 %** |
| per job | 10+ min queued | **~20 s** |

Singleton binder MSAs are not a shortcut: an empty alignment is the *correct* answer for a de novo
sequence. `N_msa 654` in the logs confirms the target's deep alignment is doing the real work.

*RING CLOSURE CONFIRMED -- this was the open question from Session 6.*

| | N-to-C distance |
|---|---|
| cyclic (10) | **1.27-1.39 A** (amide bond ~1.33 A) |
| linear (10) | 4.19-28.18 A (free termini) |

`covalent_bonds` was honoured on every cyclic design, and the linear set proves it is not applied
spuriously. **These are genuine macrocycle predictions** -- what AlphaFold Server could not have given.

*Epitope agreement: 16/20* (9/10 linear, 7/10 cyclic) land on the portal epitope BC2 designed for.

*The headline result: **r(BC2 i_pTM, Protenix ipTM) = +0.08** across 20 designs.*
**BC2's own confidence ranking does not predict whether an independent model agrees.**

- `rbp4cyc_r03` -- BC2's **best** cyclic (0.79) -> Protenix **0.326**, its lowest cyclic score.
- `rbp4lin_r02` -- BC2 0.77 -> Protenix places it on a **different site** (Jaccard 0.00, ipTM 0.38).
- `rbp4cyc_r08` -- mid-pack for BC2 (0.73) -> Protenix's **highest** (0.891).

This is the reward-hacking signature made concrete: BC2 optimises against AlphaFold2 and validates on
held-out *AlphaFold2* models -- correlated judges. **Synthesising BC2's top-ranked designs would have
included at least one that a second model says binds elsewhere.** Rank by cross-model consensus, not
by BC2 i_pTM.

*Consensus shortlist* (Jaccard >= 0.55 and Protenix ipTM >= 0.75):

| design | type | BC2 i_pTM | Protenix ipTM | Jaccard |
|---|---|---|---|---|
| `rbp4cyc_r01_l15` | cyclic | 0.77 | 0.862 | **0.91** |
| `rbp4lin_r10_l21` | linear | 0.67 | 0.801 | **0.89** |
| `rbp4cyc_r06_l16` | cyclic | 0.73 | 0.860 | 0.78 |
| `rbp4lin_r06_l24` | linear | 0.69 | 0.821 | 0.80 |
| `rbp4cyc_r08_l14` | cyclic | 0.73 | **0.891** | 0.67 |
| `rbp4lin_r03_l21` | linear | 0.75 | 0.813 | 0.73 |

*Rejected:* `rbp4cyc_r05`, `rbp4cyc_r10`, `rbp4lin_r02` (different site + low confidence);
`rbp4cyc_r04` (Jaccard 0.23); `rbp4cyc_r03`, `rbp4lin_r11` (Protenix unconfident).

**Caveats.** Agreement is evidence the pose is not an artifact, **not** evidence of affinity. Single
seed (101), 5 samples -- designs near a threshold may move. Protenix shares training data and
architectural lineage with the AF family: independent, not fully orthogonal.

**Open.** Pick 3-4 for synthesis from the consensus shortlist; optional MD triage; Protenix-v2 licence.

---

### 2026-09-23 — Session 6: both campaigns finished; linear beat cyclic; Protenix set up

**Goal.** Collect the finished campaigns, run a linear campaign for comparison, and stand up an
orthogonal structure predictor to validate the designs.

**Did.**

```bash
sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json          # job 209, linear -> 11/10 designs
uv venv --python 3.12 --managed-python --seed .venv-protenix  # separate from BC2 (JAX vs Torch)
.venv-protenix/bin/pip install --upgrade protenix             # 2.0.0
source env-protenix.sh && protenix pred -n protenix-v2 ...
```

Wrote `env-protenix.sh`, `analysis/protenix/make_inputs.py`, `analysis/alphafold_server/`.

**Found.**

*⚠️ CORRECTION to Session 4 — the recommendation to prefer cyclic was wrong for this target.*

| campaign | job | range | trajectories | accepted | **accept rate** | wall |
|---|---|---|---|---|---|---|
| cyclic | 208 | `[7,20]` | 815 | 10 | **1.2 %** | 10 h 56 m |
| **linear** | 209 | `[12,25]` | **347** | **11** | **3.2 %** | — |

**Linear was ~2.6x more efficient than cyclic**, and buried more surface (474-764 A^2 vs 371-589).
Session 4 argued for cyclic on two grounds: the design guide's *"cyclic beats linear for peptides"*,
and 6/6 linear trajectories failing at pLDDT 0.33-0.47. **The second was an artifact**: those six were
all pinned at length 12. Every one of the 11 accepted linear designs is 19-25 residues (median 23).
The variable was **length, not the ring**. The guide's claim may still hold at matched length -- we
never tested that -- but it did not hold as we applied it. Cancelling job 207 on this advice cost a
few hours of GPU time.

*Design quality, both sets.* Cyclic best is rank 3 `MHMRRPDGSEFTLEPWN` (i_pTM 0.79, pLDDT 0.88,
i_pAE 0.20, 0 off-epitope). Linear best is rank 1 `APVPKMPEEIKKEAEKWGMKMTYRP` (i_pTM 0.79, pLDDT 0.83,
BSA 764 -- the largest interface in either campaign). **Linear rank 5 binds the wrong site**
(`Hotspot_Contact_Fraction` 0.0, `Off_Epitope_Contact_Fraction` 0.62) and is excluded from validation.
Cyclic rank 9 is also suspect (0.11 / 0.20).

*Rank-1 cyclic does not exploit the empty calyx.* The Session-4 worry about AF2 being blind to retinol
was tested directly: **0 binder atoms within 4 A of the retinol cavity, nearest 9.2 A, median 15.8 A**.
It sits on the portal rim. Frames verified as comparable (1407/1407 atoms, 0.88 A without superposition).

*Apo vs holo does not matter here.* 1BRQ (apo, 2.0 A) vs 5NU7 (holo, 1.5 A), superposed: global CA RMSD
**0.53 A**; at the 10 residues rank-1 actually contacts, **mean 0.43 A, max 0.85 A** -- within
coordinate error. The one mobile outlier is **Leu35 (3.22 A)**, a hotspot no accepted design contacts;
treat a future design that leans on L35 with more suspicion. Holo is also the right state biologically
(circulating RBP4 is retinol-loaded), so **no reason to redo anything against apo**.

*AlphaFold Server: prepared but limited.* Inputs written to `analysis/alphafold_server/`. Two blockers
recorded: it is **non-commercial use only** (raised with the user; flagged for licensing sign-off), and
it **cannot express head-to-tail cyclisation** -- macrocycles would submit as linear peptides, making
low scores uninterpretable. Superseded by Protenix, which can.

*Protenix 2.0.0 installed.* Python 3.12 venv, torch 2.7.1+cu126, bf16 on the Ampere cards.
**Its input schema has `covalent_bonds`, documented for head-to-tail amide bonds** -- the reason it is
the better validator here than AlphaFold Server.

*Hardware notes worth keeping:*
- The **GTX 1660 Ti cannot run it**: compute capability 7.5 has no bf16, and bf16 is the working dtype.
  Protenix docs give 6.1 GB at 500 tokens; our jobs are ~190 tokens and measured **3.8 GB on GPU 0**,
  so two jobs per 8 GB card would fit.
- First CLI call **appeared to hang for 4 minutes**. It was `nvcc` JIT-building the `fast_layer_norm`
  kernel for five architectures (sm_80/86/89/90/100) on four cores. Cached afterwards: 3.5 s.
  `TORCH_CUDA_ARCH_LIST=8.6` is now set in `env-protenix.sh` so a rebuild targets only our cards.
- Checkpoints default to `$HOME/checkpoint`. **`PROTENIX_ROOT_DIR`** moves them; set to
  `cache/protenix_root` so weights stay in the repo like the AlphaFold params.
- The ByteDance CDN **stalled** downloading v1 (1.47 GB then 0 bytes/12 s). Partial file removed.

*protenix-v2 weights are gated.* `protenix-v2.pt` returns **HTTP 403** from the official CDN while every
v1/v0.5 checkpoint returns 200; the README calls the v2 weights "proprietary and confidential... may not
be reproduced, distributed... without express prior written consent". An **unofficial third-party
mirror** (`TMF001/protenix-v2-weights` on HuggingFace, self-labelled Apache-2.0 -- a label that
contradicts ByteDance's own terms and is not authoritative) was used **at the user's explicit
direction**, who stated the work is non-commercial. Recorded here because it is a provenance decision
someone may need to revisit, and because the licence question is unresolved rather than cleared.
Integrity was verified before use: **SHA256 matched** the published digest exactly and the checkpoint
loads to **464,442,431 parameters** (documented v2 size; v1 would be 368M). Loaded with
`weights_only=True` so an untrusted pickle could not execute code.

*MSA is a shared cost.* All 20 jobs carry the **same** RBP4 target sequence, and the binders are de novo
so their alignments are singletons regardless. The MSA server queues (`PENDING`, 60 s polls), so the
target MSA is computed **once** and reused across jobs via `unpairedMsaPath`/`pairedMsaPath` rather than
querying 20 times.

**Open.** 20 Protenix jobs; then epitope-agreement analysis.

---

### 2026-09-21 — Session 5: cyclic clears the screen gate; two Session 1 claims corrected

**Goal.** Answer "where are the logs and how far along is the campaign", and check job 208.

**Did.** Added [`progress.sh`](progress.sh) — one command for job state, counters, stage-table row
counts, screen outcomes, in-flight trajectories and GPU state:

```bash
./progress.sh                       # newest campaign folder
./progress.sh results/rbp4_cyclic   # a specific one
```

**Found.**

*The cyclic hypothesis is holding.* Trajectory 2, **`design_l16_dcfcb1293b129483` — a 16-residue
macrocycle — became the first trajectory in the project to clear the screen gate**, and then cleared
refine too:

```
passed screen design stage  i_pTM=0.54  pLDDT=0.65
passed refine design stage  i_pTM=0.69  pLDDT=0.82
```

Its `i_pTM=0.69` already exceeds the 0.6 final acceptance floor. Best of the 6 linear attempts was
i_pTM 0.24 / pLDDT 0.47 — not close.

*A clear ring-length trend, which is the actionable part:*

| length | pLDDT at screen | outcome |
|---|---|---|
| 7 | 0.35 | rejected |
| 7 | 0.37 | rejected |
| 13 | 0.37 | rejected |
| 18 | 0.58 | rejected (near miss on the 0.60 gate) |
| **16** | **0.65** | **passed screen, then refine at 0.82** |

Short rings fail; **16–18 residues is where this target works.** If yield stays low, narrowing
`binder_lengths` to roughly `[14, 20]` would spend the budget where it pays, rather than on 7–9-mers.

*⚠️ Correction to Session 1 — two claims there were wrong.* Session 1 recorded that
`1_Trajectories/!_Trajectories.csv` and `summary.csv` were **not written** for a campaign whose
trajectories all died at the screen stage, and reasoned that screen-terminated trajectories produce no
row. **Both files exist**, and `!_Trajectories.csv` *does* carry screen-terminated trajectories, with a
`terminated=screen` column recording where each stopped. `stat` shows the smoke campaign's CSV was
written at 16:12:11, one second *before* that campaign exited — so it was present when it was reported
missing. The cause was the check itself: a `[ -f … ]` test on a path containing the literal `!` in
`!_Trajectories.csv` misreported. **Use `find`, or quote the path, when testing these stage tables** —
`progress.sh` does. What Session 1 got right is that `2_Refolded/` and `3_Ranked/` are genuinely absent
until something survives the screen.

**Open.** Watch whether trajectory 2 converts into an accepted design — it still has anneal, harden and
mutate to pass, then ProteinMPNN redesign and validation. **`2_Refolded/!_Refolded.csv` will appear when
it reaches redesign**; that is the first view of `failed_filters`.

---

### 2026-09-21 — Session 4: linear launched, cancelled, switched to cyclic

**Goal.** Launch the first full campaign. Ended up switching modality mid-session; both the reasoning
and the reversal are recorded below.

**Did.**

```bash
sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json   # job 207, linear [12,25]  -> CANCELLED at 3:07
scancel 207
sbatch submit_rbp4.slurm campaigns/rbp4_cyclic.json    # job 208, cyclic [7,20]   -> RUNNING
```

**Found.**

*Job 207 (linear) started correctly* — `binder length 12 to 25, drawn per trajectory`, `i_pTM >= 0.6`,
3 workers on GPUs 0,1,2. The widened range was visibly working: design names carried their drawn
lengths (`design_l21_…`, `design_l12_…`), versus the single pinned 12 of Sessions 1/3. Cancelled after
5 trajectories (0 accepted, 2 terminated at screen). **`results/rbp4_peptide/` is preserved and
resumable** — rerun the same command to continue from trajectory 5.

*Why it was cancelled — a judgement call recorded as a mistake.* Sessions 2–3 had already established
two things pointing at cyclic: [`docs/design-guide.md`](docs/design-guide.md) states outright that
*"cyclic beats linear for peptides"* and that *"the ring's lower entropy makes it a better peptide
binder than a linear one of the same length"*; and 6/6 linear trajectories had died at pLDDT 0.33–0.47
against a 0.60 gate. Despite both, linear was recommended and launched **because it had been selected
earlier in the session** — deferring to a prior pick over the accumulated evidence. Cyclic was the
better call and should have been the recommendation. Correcting course cost ~3 min of GPU time.

*Cyclic peptide length range — the shipped sources disagree.* Check before trusting any one:

| Source | Range |
|---|---|
| `modality/cyclic_peptide.json` `binder_lengths` | **[6, 16]** ← the actual default |
| its own `description` field | "7–20 residue" |
| `docs/design-guide.md` table | "6–16 residue" |
| `examples/pdl1_cyclic_peptide.json` | **[7, 20]** |

Running at **[7, 20]** (the authors' example config, widest sanctioned span). **Cyclic's range is
narrower than linear's, not wider** — 21–25 is off-label for a macrocycle, so the linear `[12,25]`
span cannot be reproduced under cyclisation. For a strict like-for-like comparison the honest common
range is **[12, 20]**, where only the ring differs.

*Terminology, since it caused confusion:* **"macrocycle" and "cyclic peptide" are the same thing.**
There are two modalities, not three — `peptide` (linear, free termini) and `cyclic_peptide`
(head-to-tail ring). Cyclic costs no extra memory: every length 6–25 pads to the same 32-residue
bucket, so N = 206 either way.

*Job 208 startup confirmed:* `binder length 7 to 20, drawn per trajectory`, **`features cyclic peptide
closure`**, `i_pTM >= 0.6`, 3 workers on GPUs 0,1,2, all cards 82–100% at 4.5 GB.

*First cyclic results (n=2, too early to read).* Both rejections so far are **7-mers — the shortest and
least favourable ring size** — and one reached `i_pTM=0.32`, above every linear trajectory (max 0.24).
pLDDT 0.35–0.37 is still comparable to linear's 0.33–0.47, so **no clear win yet**. The longer rings
(l13, l16, l18) were still running at the time of writing; those are the informative ones.

**Open.** Job 208 running. Watch whether longer rings clear the 0.60 pLDDT gate — that is the entire
hypothesis behind the switch. Caveat standing: cyclic adds a `Cyclic_Closure_Distance` bar, so
acceptance is not strictly easier, and head-to-tail cyclisation is a synthesis route that must be
available before any cyclic design is worth ordering (`--disulfide-staple` is the guide's named
alternative rigidity route for a linear peptide).

---

### 2026-09-21 — Session 3: SLURM path validated

**Goal.** Exercise `submit_rbp4.slurm` for real, and drop the wall-clock limit.

**Did.** Set `#SBATCH --time=0` (partition `main` has `MaxTime=UNLIMITED`). Submitted the smoke
campaign to a fresh folder:

```bash
sbatch submit_rbp4.slurm campaigns/rbp4_smoke.json project_folder=results/rbp4_slurm_test
```

**Found.**

*First attempt (job 205) FAILED in 1 second* — `bindcraft design` rejected the bare
`project_folder=...` argument with a usage error. The installed console script accepts **only**
`--set KEY=VALUE`; it is **`bindcraft.py` that rewrites a bare `KEY=VALUE` into `--set KEY=VALUE`**
([`bindcraft.py:21-25`](bindcraft.py#L21-L25)). That is exactly why the shipped `bindcraft.slurm` ends
in `python bindcraft.py "$@"`. Fixed by doing the same; both spellings now work.

*Second attempt (job 206) RUNNING and correct.* Scheduled instantly — the resource request
(`cpu=4, mem=28G, gpu:4`) fits the node, confirming the Session 1 diagnosis that the shipped script's
16-core/96 GB request was the reason it would have pended. `env.sh` sourced cleanly inside the job:
venv activated, weights read from the repo cache, and GPU selection reported
`CUDA_VISIBLE_DEVICES=0,1,2,3, using GPUs 0,1,2`.

*Cosmetic wrinkle:* `bindcraft.py` prints `4 GPUs visible, designing on all of them at once` because
its banner reads `CUDA_VISIBLE_DEVICES` directly, **before** `BINDCRAFT_GPU_IDS` is applied. The actual
fan-out line is correct (`3 design workers on GPUs 0,1,2`). Ignore the banner; trust the fan-out line.

*Job 206 COMPLETED, exit `0:0`, 3 min 05 s.* The SLURM path is validated end to end.

*The compile cache is working.* 4 min 48 s cold (direct run, Session 1) vs **3 min 05 s warm** — the
`cache/compile` directory (5 entries, 9.3 MB) saved ~1.7 min on a 3-trajectory run. On a
1200-trajectory campaign that is worth having, and it is why `JAX_COMPILATION_CACHE_DIR` is pinned
into the repo rather than left to `${TMPDIR}`.

*Trajectory recipes are deterministic; their numbers are not.* Both runs drew the **same three
trajectory hashes** (`34f51ed…`, `50834f9…`, `e7b03b0…`) from the same settings, but the metrics
differed where the GPU assignment differed:

| hash | direct (i_pTM / pLDDT) | SLURM (i_pTM / pLDDT) |
|---|---|---|
| `34f51ed…` | 0.16 / 0.41 | 0.24 / 0.33 |
| `50834f9…` | 0.21 / 0.47 | 0.15 / 0.44 |
| `e7b03b0…` | 0.18 / 0.42 | **0.18 / 0.42** (same GPU both runs) |

Exactly as [`docs/design-guide.md`](docs/design-guide.md) §7 warns: *"`campaign_seed` fixes the draws
within one setup but does not guarantee identical numbers across machines/GPUs."* The one trajectory
that landed on the same card both times reproduced exactly. **Do not treat a single trajectory's
metrics as reproducible across cards** — use the `benchmark` core profile if that is ever needed.

**Evidence accumulating on the 12-mer.** Across **6/6 trajectories in two independent runs**, pLDDT
was 0.33–0.47 against a 0.60 gate, and i_pTM 0.15–0.24. Not one came close. This is no longer a
small-sample artifact: a **linear** 12-mer against the RBP4 portal looks genuinely hard, which is the
direct argument for evaluating `cyclic_peptide` — a macrocycle is conformationally constrained and
should fold far better.

**Open.** Cyclic-peptide evaluation next; then the large campaign.

---

### 2026-09-21 — Session 2: tuning the campaign for yield

**Goal.** Session 1 accepted 0/3. Work out which settings widen the search, and set a defensible
default for the first real campaign.

**Did.** Measured the cost of widening the length range; read the yield guidance in
[`docs/design-guide.md`](docs/design-guide.md) §5 and §8; rewrote `campaigns/rbp4_peptide.json`.

**Found.**

*Widening `binder_lengths` is free.* Lengths **12–32 all pad into the same 32-residue bucket**
(`padded_prediction_length(L, 32)`), so `[12,25]` compiles **one** shape and uses the identical
~4.5 GB per worker as `[12,12]` — 14 lengths for no extra cost. It also attacks the Session 1 failure
directly: a 12-mer is the hardest case in the peptide range, and pLDDT (0.41–0.47 vs a 0.6 gate) is
exactly what longer peptides improve. `[12,25]` is the `peptide` preset's own default; Session 1
narrowed it to `[12,12]` only because 12 aa was requested.

*`max_trajectories: 300` sat in a dead zone.* **`desperation_trajectories` defaults to 750** — BC2's
self-rescue ladder only engages after 750 trajectories with nothing accepted. At 300 the campaign
stopped *before BC2's own rescue mechanism could ever activate*: too many for a smoke test, too few to
trigger desperation. Fixed by raising `max_trajectories` to 1200 **and** lowering
`desperation_trajectories` to 250.

> ⚠️ Lowering `desperation_trajectories` raises the expected false-positive rate. Designs accepted on
> a ladder rung were judged against an easier task; the `autotuned` column records the rung. Weight
> them accordingly, and if a campaign produces *only* rung-6/7 designs, the epitope or the length is
> wrong — do not trust the output.

*Throughput on this box:* ~3.8 min per screen-terminated trajectory, 3 in parallel → 300 ≈ 6 h,
750 ≈ 16 h, **1200 ≈ 25 h**. Optimistic: screen-terminated trajectories are the *fast* case (50 rounds
vs 140, and no MPNN/validation). 1200 exceeds the 24 h SLURM limit, which is fine — campaigns resume
by default, so resubmit to continue.

*What deliberately was not changed.* `min_plddt_screen` stays at 0.6 and loss weights are untouched.
[`docs/design-guide.md`](docs/design-guide.md) §8 is explicit: when nothing accepts, *"adjust the
epitope, length or modality — not the loss weights."* Lowering the screen gate only pushes bad
trajectories into expensive downstream stages. `min_iptm_final` 0.7 → 0.6 was adopted — see the correction below.

*Correction to the line above (same session).* `min_iptm_final: 0.6` was first justified here as "the
authors' peptide relaxation in `examples/pdl1_peptide.json`". That is misleadingly narrow. Checking
every shipped preset and example: **22 of the 23 example campaigns set 0.6**, across *every* modality
(de novo binder, VHH, scFv, ARP, cyclic peptide, homotrimer, multidomain, humanization, IL-2R, IL-7Rα),
and `examples/dynorphin_idr.json` goes to **0.5** for a disordered target. **No shipped example runs at
the core default of 0.7**, and none of the modality presets — `peptide`, `cyclic_peptide`, `binder` —
set it at all. So 0.6 is not a peptide concession; it is the value the authors actually run campaigns
at, and the core 0.7 is a conservative headline their own configurations step down from. The setting
is better supported than first claimed, but it remains a **relaxation adopted on shipped practice, not
on our own accept-rate data** — of which we have none yet. Set it back to 0.7 to measure the true rate
at the intended bar.

*Second-order lever, noted for later:* `sequence_candidates` (10) and `kept_sequences` (1) widen MPNN
sampling per trajectory — but they are **irrelevant until trajectories survive the screen**, since
nothing has reached ProteinMPNN yet.

**Open.** Campaign not yet launched; SLURM path still unexercised.

---

### 2026-09-21 — Session 1: install, target preparation, first smoke test

**Goal.** Install BC2 from a fresh clone, keep everything self-contained in the repo, confirm the
three 8 GB cards can actually hold this target, and get one campaign to run end to end.

**Did.**

Surveyed the machine before touching the installer — which turned out to matter (see *Found*).

```bash
# 1. Environment, with uv owning the interpreter (conda base was active and had to be kept out)
uv python install 3.13                                    # uv-managed CPython 3.13.11
uv venv --python 3.13 --managed-python --seed .venv       # pre-built so install.sh adopts it
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" bash install.sh

# 2. Move the 5.3 GB weight cache out of ~/.cache and into the repo
mv ~/.cache/bindcraft/alphafold cache/alphafold           # env.sh points BINDCRAFT_WEIGHTS here

# 3. Target
curl -sf -o targets/5NU7.pdb https://files.rcsb.org/download/5NU7.pdb
awk '$1=="ATOM" && substr($0,22,1)=="A" && (substr($0,23,4)+0)<=174' targets/5NU7.pdb > targets/RBP4.pdb
printf 'TER\nEND\n' >> targets/RBP4.pdb

# 4. Smoke test: 1 design, 3 trajectories
source env.sh && bindcraft design campaigns/rbp4_smoke.json
```

Wrote `env.sh`, `campaigns/*.json`, `submit_rbp4.slurm`, `.gitignore`, this logbook.

**Found.**

*Machine — three things that differed from the brief ("3× RTX 3070"):*

1. **Four GPUs, not three, and mixed.** GPU 3 is a **GTX 1660 Ti, 6 GB, compute cap 7.5** alongside
   three 8 GB Ampere cards. SLURM advertises a flat `gpu:4` and will hand it out.
2. **The shipped `bindcraft.slurm` cannot schedule on this node, and fails silently.** It sizes itself
   at 4 cores + 24 GB *per GPU* (→ 16 cores / 96 GB for 4 GPUs); its `#SBATCH` fallbacks are 8 cores /
   48 GB. The node has **4 cores / 31 GB total**, so all three figures exceed it and the job would
   **pend indefinitely rather than error**. Replaced by `submit_rbp4.slurm`.
3. **`gres.conf` declares GPUs with no `Type=` field**, so SLURM cannot be asked for "a 3070" and
   `--gres=gpu:3` may hand over the 1660 Ti. Worked around by requesting all 4 and selecting cards by
   VRAM in `env.sh`.

*GPU memory — the main risk, and it cleared.* BC2 budgets `2.0 × (3.4 GB + 38 kB × N²)` per worker
([`design_workers.py:14-16`](bindcraft/design_workers.py#L14-L16)). For N = 174 + 32 = 206 that is
**10.6 GB, more than an 8 GB card**, and it announces exactly that:
`campaign fan-out: 3 design workers on GPUs 0,1,2 at 10.6 GB each`. It does not refuse —
[`design_workers.py:70`](bindcraft/design_workers.py#L70) clamps to `max(1, …)`. **Measured actual use:
4.5 / 2.4 / 4.5 GB.** The 2.0× safety factor is conservative and 8 GB is genuinely sufficient here.

*Parallelism needed no work.* BC2 fans out one worker per visible card, each running independent
trajectories — data-parallel over design attempts, not a sharded model. Observed all three GPUs at
62–100% simultaneously.

*Smoke test: ran correctly, accepted nothing.* All 3 trajectories were **rejected at the screen stage
on pLDDT**:

| trajectory | worker/GPU | i_pTM | pLDDT | outcome |
|---|---|---|---|---|
| 1 `34f51ed…` | 2 | 0.16 | 0.41 | rejected at screen `[pLDDT]` |
| 2 `50834f9…` | 0 | 0.21 | 0.47 | rejected at screen `[pLDDT]` |
| 3 `e7b03b0…` | 1 | 0.18 | 0.42 | rejected at screen `[pLDDT]` |

The gate is `min_plddt_screen = 0.6` (core default). Stage budget is
`screen 50 → refine 25 → anneal 45 → harden 5 → mutate 15`
([`settings.py:604`](bindcraft/settings.py#L604)), so all three died at the **first** gate, after 50
rounds, and never reached ProteinMPNN. Hence **stages `2_Refolded/` and `3_Ranked/` were never created**,
and `summary.csv` was not written — [`campaign_output.py:445`](bindcraft/campaign_output.py#L445)
skips it when there are no completed trajectory rows. `.campaign_state.json` recorded the run properly
(`trajectories: 3, accepted: 0`) and the process exited cleanly.

**This is a normal outcome at n=3, not a failure.** A linear 12-mer is largely disordered when free, so
low pLDDT is intrinsic to the modality. The machinery is validated; the yield question is still open.

*Two false starts, recorded so they are not repeated:*

- I briefly assumed 5NU7 was the only gap-free structure of the three. **Wrong** — 1RBP and 3FMZ are
  gap-free too. The real differentiators are resolution and chain count (see [Reference](#target-provenance)).
- I deleted `rbp4_smoke.json` from the repo root *while the campaign was running*, when moving the
  campaign files into `campaigns/`. **Harmless, as it turned out**: the workers had already finished at
  16:11:45 / 16:11:59 / 16:12:11, past the 16:10:55 deletion, because workers read
  `workers/campaign_settings.json` rather than the original file. Do not rely on this.

**Open.** SLURM path unexercised; stages 2–3 unexercised; peptide accept rate unknown.

---

## Open items

- [x] ~~Run the 20 Protenix jobs and analyse epitope agreement~~ — **done, Session 7**: 16/20 agree.
- [x] ~~Verify the ring actually closed~~ — **confirmed**: 1.27–1.39 Å across all 10 cyclics.
- [ ] **Pick the final 3–4 for synthesis** from the **9 designs** that pass the stability gates
      (Session 8 table, `analysis/protenix/agreement.csv`). The compute side is done; what is missing
      is non-computational — macrocycle synthesis feasibility and the licence question below.
      Rank by **cross-model consensus, never BC2 i_pTM** (r = +0.06, and −0.19 against jaccard).
      Suggested four: `rbp4lin_r09_l19`, `rbp4lin_r06_l24`, `rbp4lin_r10_l21`, `rbp4cyc_r08_l14`.
- [x] ~~Reconsider the two knife-edge exclusions~~ — **resolved, Session 8**: `rbp4lin_r09_l19` was
      excluded by a two-sided sd gate despite hitting jaccard 1.00 in 3/5 seeds — **admitted**.
      `rbp4cyc_r09_l20` is stable at 0.44–0.60 throughout — genuinely borderline, keep as backup.
- [ ] **Make the jaccard stability gate one-sided** in `analyse.py` (or gate on `jaccard_min` alone).
      As written it penalises designs whose agreement varies *upward*. Currently applied by eye.
- [x] ~~Consider a second Protenix seed for the shortlist~~ — **done, Session 8**: 5 seeds
      (101/202/303/404/505). Two verdicts flipped. Reseeding costs 18 min, so treat it as routine.
- [ ] **Optional MD triage** — now worth aiming at the **9 survivors** rather than all 20:
      1 rep × 50 ns (~1.5 days on 3 GPUs), then 3 reps × 200 ns on whatever holds up. Use CHARMM36m force-switch settings; build cyclic topologies via CHARMM-GUI
      (`pdb2gmx` will wrongly add charged termini to a macrocycle).
- [ ] **Resolve the Protenix-v2 licence question** with whoever handles licensing — weights came from
      an unofficial mirror of a gated, "proprietary and confidential" checkpoint.
- [ ] **Confirm the synthesis route** for head-to-tail macrocycles before ordering any cyclic design.
- [ ] Ask the admin for `Type=` in `gres.conf` (`Type=rtx3070` / `Type=gtx1660ti`), so
      `--gres=gpu:rtx3070:3` works and the request-4-filter-3 workaround can go.
- [ ] More designs are cheap if wanted — linear ran at 3.2 %, so ~30 more is a few hours.
- [ ] A third architecture (Boltz-2 / Chai-1) is **only worth it if BC2 and Protenix disagree**.
- [ ] Move `results/rbp4_cyclic/binder_location.png` into `analysis/` — it is my diagnostic plot, not a
      BC2 output, and `bindcraft archive` would sweep it up.

## Reference

*Durable facts. Edit in place; do not append here.*

### Hardware

| idx | card | VRAM | compute cap |
|----|------|------|-------------|
| 0 | RTX 3070 Ti | 8 GB | 8.6 |
| 1 | RTX 3070 | 8 GB | 8.6 |
| 2 | RTX 3070 | 8 GB | 8.6 |
| 3 | **GTX 1660 Ti** | **6 GB** | 7.5 — excluded by `env.sh` |

Node `AIMecoSPECS`, partition `main`, **4 CPUs / 31000 MB RAM**, `Gres=gpu:4`, Slurm 23.11.4.
Driver 590.48.01 / CUDA 13.1. 656 GB free on `/`. **CPU and RAM are the binding constraint, not VRAM.**

### Paths and environment

Everything is kept inside the repo. **`source env.sh` at the start of every session** — it activates
the venv and sets:

| var | path | holds |
|-----|------|-------|
| `BINDCRAFT_WEIGHTS` | `cache/` | AlphaFold params → `cache/alphafold` (5.3 GB, 7 models) |
| `JAX_COMPILATION_CACHE_DIR` | `cache/compile` | compiled XLA graphs (~60 s to build, 1.5 s to reload) |
| `TMPDIR` | `tmp/` | scratch |
| `BINDCRAFT_GPU_IDS` | `0,1,2` | **selected by VRAM ≥ 8000 MiB, not hardcoded** — keeps the 1660 Ti out |

ProteinMPNN weights ship inside the package (`bindcraft/weights/proteinmpnn/`); nothing to download.
**The install is editable — moving or renaming this directory breaks the `bindcraft` command.**

### Layout

| path | what |
|---|---|
| `campaigns/` | campaign JSONs (**tracked**) |
| `targets/` | downloaded + prepared structures (ignored; regenerate with the commands above) |
| `cache/`, `tmp/`, `results/`, `.venv/` | generated (ignored) |
| `env.sh`, `env-protenix.sh`, `submit_rbp4.slurm`, `progress.sh` | site config (**tracked**) |
| `run_protenix.sh`, `run_protenix_seeds.sh` | Protenix validation and reseeding (**tracked**) |
| `.gitignore`, `LOGBOOK.md` | repo hygiene and this record (**tracked**) |

> **Path gotcha:** `target_path` is resolved relative to the **campaign JSON's own directory**
> ([`settings.py:710-711`](bindcraft/settings.py#L710-L711)) — hence `../targets/RBP4.pdb`.
> `project_folder` is *not* rewritten and stays relative to the working directory, so
> **always submit from the repository root.**

### Commands

```bash
source env.sh
bindcraft design campaigns/rbp4_smoke.json            # 1 design / 3 trajectories, direct
bindcraft design campaigns/rbp4_peptide.json          # 10 designs / 300 trajectories, direct
sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json  # via SLURM (use THIS, not bindcraft.slurm)

bindcraft rank results/rbp4_peptide --on i_pTM        # re-rank without redesigning
bindcraft filter results/rbp4_peptide                 # which criteria rejected what
```

Campaigns **resume by default** — rerun against the same `project_folder` with the same flags.

### Checking on a running campaign

```bash
./progress.sh                       # newest campaign folder
./progress.sh results/rbp4_cyclic   # a specific one
```

| Where | What it holds |
|---|---|
| `bindcraft_<jobid>.out` (repo root) | SLURM job stdout: preflight, fan-out, and trajectories **in order** |
| `<project>/workers/worker_NN_gpu_N.log` | **each worker's live record, written line by line** — read this for what is happening *now* |
| `<project>/.campaign_state.json` | counters: trajectories, accepted, terminated, failed filters |
| `<project>/1_Trajectories/!_Trajectories.csv` | one row per finished trajectory, incl. screen-terminated, with `terminated` |
| `<project>/2_Refolded/!_Refolded.csv` | scored ProteinMPNN candidates + `failed_filters` — **appears only once something survives the screen** |
| `<project>/3_Ranked/!_Ranked.csv` | **the result**: accepted designs, best first |

The console log holds a finished trajectory until no worker is still designing a lower-numbered one,
so it can lag; **the per-worker logs are the live view.**

> Stage tables are named `!_*.csv`. A `[ -f … ]` test on that literal `!` misreports — use `find` or
> quote the path. See the Session 5 correction.

### Orthogonal validation (Protenix)

```bash
source env-protenix.sh          # separate venv: BC2 is JAX, Protenix is Torch
protenix pred -i analysis/protenix/rbp4cyc_input.json -o analysis/protenix/out_cyclic \
  -n protenix-v2 --use_default_params True --use_msa True --msa_server_mode protenix
```

| | |
|---|---|
| Model | `protenix-v2`, 464,442,431 params, bf16, cycle=10 step=200 sample=5 |
| Weights | `cache/protenix_root/checkpoint/protenix-v2.pt` (via `PROTENIX_ROOT_DIR`) |
| Memory | ~3.8 GB at ~190 tokens → two jobs fit per 8 GB card |
| Inputs | `analysis/protenix/make_inputs.py` regenerates all JSONs |
| **Cyclisation** | `covalent_bonds`: N of residue 1 ↔ C of residue L, both on entity `"2"` |

**Adding seeds** (for error bars — see Session 8):

```bash
./run_protenix_seeds.sh                 # default seeds 202,303,404,505
./run_protenix_seeds.sh 606,707         # or any comma-separated list
```

**~29 s of GPU per job-seed**, plus a ~100 s one-time model load; 80 job-seeds on two cards took
**17 m 43 s**. **No MSA server is involved** — the input JSONs carry absolute `pairedMsaPath` /
`unpairedMsaPath` into `msa_cache`. Protenix writes `<job>/seed_NNN/` alongside existing seeds, so
reseeding is additive and overwrites nothing.

> **`--sample 5` is not a source of diversity.** The five samples within one seed differ by ~0.005 in
> ipTM — they are near-duplicates. Between seeds the shift is ~5× larger. **Widen with seeds.**

| file | what |
|---|---|
| `analysis/protenix/agreement.csv` | **the ranking table** — one row per design, mean ± sd across seeds |
| `analysis/protenix/agreement_per_seed.csv` | one row per design-seed, for checking a specific draw |
| `analysis/protenix/agreement_seed101.csv` | frozen Session 7 single-seed table, kept for comparison |

> **`analyse.py` must group by seed.** It picks the most confident sample *within* each seed
> ([`best_sample()`](analysis/protenix/analyse.py#L87)) and then averages *across* seeds. Picking the
> best sample across all seeds — which a bare `job.rglob('*.cif')` does — reports a maximum that
> grows with every seed added. See Session 8.

> **Counting seed directories:** they are at depth 3, `out/<set>/<job>/seed_NNN`. A
> `find -maxdepth 2` silently returns nothing and looks exactly like a failed run.

> **The GTX 1660 Ti cannot run Protenix** — cc 7.5 has no bf16, and bf16 is the working dtype.
> **`TORCH_CUDA_ARCH_LIST=8.6`** avoids a 4-minute five-architecture JIT kernel rebuild.

### Target provenance

**PDB 5NU7**, human retinol-binding protein 4.

| | 1RBP | **5NU7** | 3FMZ |
|---|---|---|---|
| Resolution | 2.00 Å | **1.50 Å** | 2.90 Å |
| Chains | A | **A** | A + B |
| Ligand | retinol | **retinol** | 2T1 (synthetic) |
| Chain breaks | none | none | none |

Chosen on resolution (interface design is built on these coordinates, and the epitope core Trp67/Phe96
are exactly the large side chains whose rotamers need to be well determined), single chain, and native
ligand. *1RBP is a perfectly usable fallback — the margin is modest.*

**Preparation:** chain A only; retinol/Cl⁻/waters stripped (BC2's AlphaFold pipeline has no parameters
for a retinol cofactor); **Asp175 dropped — disordered, only its backbone N present.** Result: residues
1–174, 1407 atoms, no altlocs, no insertion codes, standard residues only. Disulfides intact:
Cys4–Cys160, Cys70–Cys174, Cys120–Cys129; no free cysteines.

### Epitope: the retinol calyx portal

RBP4 is a lipocalin — an 8-stranded β-barrel holding retinol in a cup (the *calyx*). The mouth of that
cup is the **portal**: where retinol enters and leaves, overlapping the transthyretin (TTR) binding
surface. It is the pharmacologically meaningful site (what A1120 and fenretinide act on).

Selected in two steps: residues within 5 Å of retinol, then filtered by solvent accessibility
(Shrake–Rupley vs Tien et al. maxima) — barrel-interior residues contact retinol but are **unreachable
by a peptide**. Exposed rim (rel. SASA): Ser95 74%, Phe96 66%, Leu64 66%, Lys99 58%, Val93 56%,
Trp67 55%, Leu35 40%, Arg62 35%, Gln98 34% — a hydrophobic groove framed by charges.

**`hotspots = "35,62,64,67,93,95,96,98,99"`**, soft targeting (no `--forced-targeting`).
*Cys70 is exposed (31%) but excluded — it is locked in the Cys70–Cys174 disulfide.*

> Without hotspots BC2 puts the binder wherever AF2 most easily makes a confident interface — often a
> crystal-packing face or a site occluded in vivo. Good scores, no biological meaning.

### Design settings in play

`modality: peptide`, `binder_lengths: [12, 25]` (Session 2; Session 1 used `[12,12]`). An earlier plan
for 10 aa was dropped as *below* the preset's validated range. **Lengths 12–32 share one 32-residue
pad bucket, so widening within that span is free** — one compiled shape, identical memory.

Active filters: `Backbone_Clashes ≤ 0`, `Interface_Residues ≥ 7`, `i_pAE ≤ 0.33`, `i_pTM ≥ 0.7`.
Stage gates from core defaults: `min_plddt_screen 0.6`, `min_plddt_refine 0.6`, `min_iptm_anneal 0.5`.
The `peptide` preset nulls the `pTM` and `Unbound_Binder_pLDDT` filters and sets `max_ipae_final 0.33`,
`refine_steps 75`.

### If it runs out of GPU memory

Cheapest first: `workers_per_gpu=1` → `subbatch_size=4` → smaller `binder_lengths` → larger
`length_bucket_size`. Note auto-chunking only engages above **384 residues**
([`af2.py:203`](bindcraft/af2.py#L203)), so at N=206 this campaign runs unchunked.
