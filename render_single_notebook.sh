#!/bin/bash

# Install missing dependencies for notebook rendering

CONFIG_PATH="$1"

# Extract output_root using python
OUTDIR=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg.get('output_root') or cfg.get('paths', {}).get('output_root', ''))")

if [ -z "$OUTDIR" ]; then
    echo "No output_root found in $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$OUTDIR" ]; then
    echo "Output directory does not exist: $OUTDIR"
    exit 1
fi

echo "Output directory: $OUTDIR"

# List of notebooks to render
NOTEBOOKS=("deterministic_verification.ipynb" "probabilistic_verification.ipynb")

for nb_name in "${NOTEBOOKS[@]}"; do
    nb="notebooks/${nb_name}"
    if [ ! -f "$nb" ]; then
        echo "Notebook not found: $nb"
        continue
    fi

    out_nb_path="${OUTDIR}/${nb_name}"
    echo "Rendering $nb_name..."

    # Execute notebook with papermill
    # We pass the absolute path of the config
    if ! python -m papermill "$nb" "$out_nb_path" -p config_path_str "$CONFIG_PATH"; then
        echo "ERROR: Failed to render $nb_name"
        continue
    fi

    echo "Converting to HTML..."
    if ! python -m jupyter nbconvert --to html "$out_nb_path"; then
        echo "ERROR: Failed to convert $nb_name to HTML"
    fi
done
