# AtlanticBreeze

Eastern North Atlantic cloud simulations with Breeze, grounded in ARM observations and regional reanalysis.

The first case is the **March 4, 2016 closed-cell stratocumulus** studied by [Ghate and Cadeddu (2019)](https://doi.org/10.1029/2018JD030141). We collect February 26–March 11 to examine surrounding conditions and select suitable simulation intervals.

## Keep it small

- `cases/march2016.toml`: dates and geographic bounds.
- `scripts/download_data.py`: ARM, ERA5, and MERRA-2 acquisition.
- `data/` and `outputs/`: local, ignored by Git.

This is a case-study repository, not a replacement for Breeze or a general experiment framework. Simulation drivers and Julia figures will be added when needed. No runnable simulation is included yet.

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
