# AtlanticBreeze

Eastern North Atlantic cloud simulations with Breeze, grounded in ARM observations and regional reanalysis.

The first case is the **March 4, 2016 closed-cell stratocumulus** studied by [Ghate and Cadeddu (2019)](https://doi.org/10.1029/2018JD030141). We collect February 26–March 11 to examine surrounding conditions and select suitable simulation intervals.

## Keep it small

- `cases/march2016.toml`: dates and geographic bounds.
- `scripts/download_data.py`: ARM, ERA5, and MERRA-2 acquisition.
- `data/` and `outputs/`: local, ignored by Git.

This is a case-study repository, not a replacement for Breeze or a general experiment framework. The forward model, observation plots, and forcing-training entry points live in `scripts/`. The one-interval workflow is described below; full-interval validation is still required.

## Download data

Use Python 3.11 or newer and install the acquisition dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py arm
python scripts/download_data.py era
python scripts/download_data.py merra
```

Each command resumes existing completed files. Use `--data-dir /path/to/storage` for external storage and `--case cases/march2016.toml` to select a configuration. Run at most one collector per source and output directory. The collector keeps a 25 GiB free-space reserve.

Credentials remain outside Git:

- **ARM:** `ARM_USERNAME` and `ARM_TOKEN` environment variables.
- **ERA5:** the CDS client's standard configuration, with the relevant dataset terms accepted.
- **MERRA-2:** Earthdata credentials in the standard `.netrc`, with GES DISC access enabled. `curl` is required; the collector creates a private session-cookie file in the data directory.

Successful files are opened as NetCDF and logged with SHA256 checksums in `files.jsonl`. Request settings, ARM inventories, and sanitized failures are recorded separately. A collector reaching the end does **not** establish complete coverage: inspect errors and inventories. Missing observations must never be interpreted as clear sky.

## Data scope

ARM: soundings, microwave LWP, cloud boundaries, ceilometer, meteorology, aerosol/CCN, radar moments, and Meteosat VISST products where available. ERA5: hourly surface and pressure-level fields over a 6° square around ENA. MERRA-2: hourly surface/flux/radiation products and three-hourly atmospheric profiles, cropped to the same region. Temporary global MERRA-2 downloads are removed after validated regional subsets and source checksums are saved.

The paper's scattering-aware LWP and drizzle retrievals are not automatically equivalent to standard ARM products. VISST retrievals are also distinct from the paper's SEVIRI visible imagery. Availability and comparison operators require verification. Pressure-level reanalysis supports initial forcing estimates; observed soundings are the primary source for boundary-layer structure.

## Initial study plan

1. Inspect surrounding days using LWP, cloud structure, soundings, and satellite context.
2. Select a 6–12-hour experiment with an explicit spin-up period.
3. Establish a Breeze baseline with prognostic aerosols, then compare fixed aerosols.
4. Constrain forcing before jointly exploring forcing and microphysics sensitivity.

Raw measurements remain unchanged. Smoothing, forcing reconstruction, and evaluation masks will be explicit derived products. The earlier July 2017 ENA and CASS studies remain separate.

## One sounding interval

The first interval is **2016-03-04 11:30–23:31 UTC** (43,260 s). Preparation requires the March 4 and March 5 ERA5 files to bracket the final observation in time:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/prepare.jl
julia --project=. scripts/forward.jl
julia --project=. scripts/train.jl
```

`prepare.jl` uses the active Python environment for extraction and Julia for plotting. Set `ATLANTIC_DATA` for raw data and `ATLANTIC_PYTHON` if its interpreter is elsewhere.

`forward.jl` and `train.jl` share `model.jl`; set `ATLANTIC_INPUT` and `ATLANTIC_OUTPUT` to override their directories. Output directories must be empty. `ATLANTIC_NX` defaults to 128; the horizontal spacing is always 100 m. The 12.8 km periodic domain is an inexpensive starting candidate, **not an established minimum or representation of all mesoscale weather**. The vertical grid has 25 m cells below 2.5 km and 100 m above, to 8 km (155 cells). The upper 20% damps velocity. Domain-size sensitivity remains necessary.

The model uses P3 with the adapter's prognostic `aer2` aerosol distribution, ecCKD, Smagorinsky closure, interactive ocean fluxes driven by ERA5 SST, and a fixed 1 s timestep. There is no LASSO diagnostic aerosol-reservoir reset: aerosol evolves prognostically. The aerosol distribution is an initial assumption, not a fit to March CCN observations. Float64 is used for finite-difference sensitivity. The pinned Breeze example helpers supply physical operators; its July input files and its `build_case` function are not used.

Preparation preserves raw inputs and records checksums. It averages sonde measurements over 50 m windows, retaining the observed inversion, and derives humidity from dewpoint. It does not artificially homogenize the free troposphere. Initialization is condensate-free; a one-hour spin-up exclusion is explicit. ERA5 horizontal advection is calculated on pressure surfaces: theta advection is converted to a temperature tendency with Exner before the energy forcing is applied. Subsidence is separate, avoiding double counting. Geostrophic winds come from geopotential gradients. This initial estimate uses the nearest ERA5 column and central horizontal differences; it is not optimized using the final sounding.

### Evaluation and training

The forward script saves LWP, final profiles, run settings, and `observation_error.png`. The loss is the sum of:

- Mean squared error of post-spin-up 10-minute LWP, scaled by 50 g m⁻².
- Final dry-potential-temperature profile error below 2.5 km, scaled by 1 K.
- Final vapor profile error below 2.5 km, scaled by 1 g kg⁻¹.

These are explicit balancing scales, not asserted observational uncertainties. LWP requires good `qc_phys_lwp`, good `phys_qc_flag`, convergence, at least 10 valid samples and at least 50% valid coverage per bin. Missing bins are omitted, never filled with zero. Sonde ascent time and point-versus-domain sampling remain comparison limitations. Training performance on this interval is not out-of-sample validation.

The initial training script uses **central finite differences and backtracking**, not automatic differentiation. It fits two vertically tapered corrections: temperature advection in K/day and vapor advection in g/kg/day, bounded by ±3 and ±2 respectively. Each evaluation rebuilds the same model and deterministic perturbations. It defaults to one optimization iteration (`ATLANTIC_TRAIN_ITERATIONS`), accepts only a lower loss, and saves every evaluation and the fitted controls. This conservative baseline can later be replaced with a validated adjoint without changing the observation loss. It does not fit microphysics yet.

### GCP

From a dedicated checkout after instantiating its Julia environment:

```sh
export ATLANTIC_INPUT=/path/to/prepared/interval
# Engineering test at the intended horizontal grid; not an observation comparison.
ATLANTIC_STOP_SECONDS=2 sbatch scripts/gcp.sbatch forward
# Full interval:
sbatch scripts/gcp.sbatch forward
# Training costs several full forward evaluations:
sbatch --job-name=atlantic-train scripts/gcp.sbatch train
```

`ATLANTIC_CPU=true ATLANTIC_NX=8 ATLANTIC_STOP_SECONDS=2` provides a small CPU engineering test. Such tests deliberately report no final-sounding loss. No restart/pickup support is claimed yet; interrupted trials should use a fresh output directory. Scheduler logs remain outside Git.

### Current validation

Input extraction and the Julia observation/forcing plot have been exercised on the March 4 data. An 8² CPU engineering test advanced two timesteps with finite LWP and final profiles. The optimizer control flow passed a synthetic quadratic check; this is not a physical gradient test. The 128² GPU smoke test and full-interval validation are still pending GPU capacity. No trained LES result is claimed yet.
