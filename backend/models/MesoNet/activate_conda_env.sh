# Assuming the Conda environment is located on Conda's default created path
CONDA_PATH="$HOME/miniconda3/bin/conda"

# If Conda path is executable
if [ -x "$CONDA_PATH" ]; then
    eval "$("$CONDA_PATH" shell.bash hook)"
    # If the environment is not found, then Conda will output the error
    conda activate mesonet
else
    echo "Error: Conda not found at $CONDA_PATH"
fi
