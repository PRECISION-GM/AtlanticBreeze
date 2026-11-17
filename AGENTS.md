# Working agreements

- Keep this repository focused on Eastern North Atlantic cloud cases with Breeze.
- Use Julia for simulations, analysis figures, and animations. Python is acceptable for data acquisition and inspection.
- Prefer a few readable scripts and case configurations over a framework.
- Keep credentials, raw observations, model output, and movies out of Git.
- Record data sources, retrieval settings, checksums, and QC choices; distinguish observations from derived estimates.
- Preserve existing case results when adding experiments. Label fixed versus prognostic aerosol configurations explicitly.
- Do not copy the BreezeLab repository or unrelated experiments into this project.

# Julia style

- Follow the Oceananigans and Breeze examples: explain the experiment, then present the grid,
  reference state, forcing, boundary conditions, model, initialization, and simulation in that order.
- Use descriptive names for models, checkpoint state, adjoints, inputs, and output records.
  Reserve short mathematical names for local equations where their meaning is clear.
- Put independent statements on separate lines. Avoid semicolon-packed setup and control flow.
- Use spaces around assignments and keyword arguments; wrap long constructors vertically.
- Explain physical units and non-obvious numerical requirements beside the relevant code.
  Keep job histories and debugging narratives in reports, not in the main simulation path.
- Keep refactors separate from numerical changes. Check parsed syntax equivalence for formatting
  changes and run Julia validation only on allocated compute nodes.
