# Installation & Quickstart

## Quickstart (default: container with Podman + Enroot)

1. Clone into the expected CSCS path (important)

   Several paths in the provided `tools/edf_template.toml` as well as example YAML configs assume the repository lives under:

   ```bash
   /capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
   ```

   If you place the repo elsewhere, you must adapt those absolute paths (image, workdir, mounts) in your generated EDF TOML and any YAML configs that reference model/data locations. To follow the convention used by collaborators, clone like this:

   ```bash
   mkdir -p /capstor/store/cscs/swissai/a122/$USER
   git clone git@github.com:swiss-ai/SwissClim_Evaluations.git /capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
   cd /capstor/store/cscs/swissai/a122/$USER/SwissClim_Evaluations
   ```

   We recommend the container workflow for fastest, reproducible setup.

1. Build the container (Podman) at repo root in an interactive session:

   ```bash
   srun --container-writable -t 01:00:00 -A a122 -p debug --pty bash
   podman build -t swissclim-eval .
   ```

1. (CSCS Alps) Export to Enroot SquashFS and set up EDF once:

   ```bash
   rm -f tools/swissclim-eval.sqsh
   enroot import -x mount -o tools/swissclim-eval.sqsh podman://swissclim-eval
   exit # exit the interactive build session
   mkdir -p ~/.edf
   sed "s/{{username}}/$USER/g" tools/edf_template.toml > ~/.edf/swissclim-eval.toml
   ```

1. Review and edit the example config:

   The project ships with a commented config that explains every key and valid
   values. Copy it and adjust the paths and selections as needed.

   ```bash
   cp config/example_config.yaml config/my_run.yaml
   ```

1. (CSCS Alps) Launch an interactive session using the container:

   ```bash
   srun --container-writable --environment=swissclim-eval -A a122 -t 01:30:00 -p debug --pty /bin/bash
   ```

   You are now inside the container with all dependencies installed.
   For a richer debugging experience we recommend using `code tunnel`.

1. Run:

   ```bash
   python -m swissclim_evaluations.cli --config config/my_run.yaml
   ```

   Outputs appear under paths.output_root (one sub-folder per module).

   Note: For reproducibility, the CLI copies the exact YAML config you pass with `--config`
   into the output_root directory at the start of the run (using the original filename).

1. Or submit a batch job (CSCS Alps):

   ```bash
   sbatch launchscript_single.sh
   ```

   Don't forget to adjust the path to your `config/my_run.yaml` in
   `launchscript_single.sh` if you placed it elsewhere.
1. Run multiple configs in one batch allocation (CSCS Alps):

   Populate `eval_configs.txt` with one config path per line (relative paths are supported), then submit:

   ```bash
   sbatch launchscript_multi.sh
   ```

   Notes:
   - `launchscript_multi.sh` reads `eval_configs.txt` and spawns one task per config via `srun --multi-prog`.
   - Keep `#SBATCH --ntasks-per-node` aligned with the number of configs you want to run concurrently.
   - Each task writes its own logs under `logs/` and outputs under each config's `paths.output_root`.
1. Here is a one-liner with `srun` instead of the `launchscript_single.sh`:

   ```bash
   srun --job-name=swissclim-eval --time=01:30:00 --account=a122 --partition=normal --container-writable --environment=swissclim-eval /bin/bash -c 'export PYTHONUNBUFFERED=1 && python -u -m swissclim_evaluations.cli --config config/my_run.yaml'
   ```

> Prefer a plain virtual environment? Use one of the alternatives below.

### Install with uenv + uv

```bash
bash tools/setup_env_uenv.sh # activates uenv and exits
bash tools/setup_env_uenv.sh # installs deps with uv
# Activates .venv and installs deps via uv
python -m swissclim_evaluations.cli --config config/my_run.yaml
```

### Install with conda + uv

```bash
bash tools/setup_env_conda.sh
conda activate swissclim-eval
python -m swissclim_evaluations.cli --config config/my_run.yaml
```
