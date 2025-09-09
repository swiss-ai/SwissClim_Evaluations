#!/bin/bash
#SBATCH --job-name=swissai-eval
#SBATCH --output=logs/swissai%j.out
#SBATCH --error=logs/swissai%j.err
#SBATCH --time=12:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Run with SLURM CPU binding
srun \
  --cpu-bind=cores \
  --container-writable \
  --environment=/capstor/store/cscs/swissai/a122/sadamov/SwissClim_Evaluations/tools/swissai_container.toml \
  python -m swissclim-evaluations --config /capstor/store/cscs/swissai/a122/sadamov/SwissClim_Evaluations/configs/eval_config.yaml
