#!/bin/bash

# Script: run_all.sh
# Purpose: Executes three related commands in sequence with smart argument passing.

# --- Script Description ---
#
# This script accepts arguments and passes them to the following three commands:
# 1. mpirun -np <NP_VAL> ./queryer --load <LOAD_DIR> --smpl <SMPL_VAL> [OTHER_ARGS]
# 2. ./optimizer --load <LOAD_DIR>
# 3. mpirun -np <NP_VAL> ./queryer --load <LOAD_DIR> [OTHER_ARGS]
#
# - The `--load` argument's value is passed to all three commands.
# - The `--smpl` argument's value is passed only to the first command.
# - The `--np` argument's value sets the number of processes for mpirun (defaults to 4).
# - All other arguments ([OTHER_ARGS]) are passed to the first and third commands.

# --- Argument Initialization ---
LOAD_DIR=""
SMPL_VAL=""
NP_VAL="4" # Default value for the number of processes
OTHER_ARGS=() # Array for other arguments

# --- Parse Incoming Arguments ---
# Use a while loop to process all arguments passed to the script
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --load)
      LOAD_DIR="$2"
      shift # shift past --load
      shift # shift past its value
      ;;
    --smpl)
      SMPL_VAL="$2"
      shift # shift past --smpl
      shift # shift past its value
      ;;
    --np)
      NP_VAL="$2"
      shift # shift past --np
      shift # shift past its value
      ;;
    *)
      # If it's not a recognized argument, treat it as an "other" argument
      OTHER_ARGS+=("$1")
      shift # shift past this argument
      ;;
  esac
done

# --- Check for Required Arguments ---
# Check if the --load argument was provided, as it's required by all commands.
if [ -z "$LOAD_DIR" ]; then
  echo "Error: The --load argument is required."
  echo "Usage: $0 --load <directory> --smpl <value> [--np <number>] [other_queryer_args]"
  exit 1
fi
if [ -z "$SMPL_VAL" ]; then
  echo "Error: The --smpl argument is required for the first command."
  echo "Usage: $0 --load <directory> --smpl <value> [--np <number>] [other_queryer_args]"
  exit 1
fi

# --- Build and Execute Commands ---

# Set a flag to track if any command fails
has_failed=false

# **Command 1: queryer (with --smpl and other args)**
echo "--- (1/3) Running the first queryer command... ---"
CMD1="mpirun -np ${NP_VAL} ./queryer --load ${LOAD_DIR}"
if [ -n "$SMPL_VAL" ]; then
  CMD1+=" --smpl ${SMPL_VAL}"
fi
# Add other arguments
if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
    CMD1+=" ${OTHER_ARGS[@]}"
fi

echo "Executing: $CMD1"
eval $CMD1
if [ $? -ne 0 ]; then
    echo "Error: Command 1 failed."
    has_failed=true
fi
echo "--- Command 1 finished ---"
echo

# **Command 2: optimizer**
# Only execute if the previous command succeeded
if [ "$has_failed" = false ]; then
    echo "--- (2/3) Running optimizer... ---"
    CMD2="./optimizer --load ${LOAD_DIR}"
    echo "Executing: $CMD2"
    eval $CMD2
    if [ $? -ne 0 ]; then
        echo "Error: Command 2 failed."
        has_failed=true
    fi
    echo "--- Command 2 finished ---"
    echo
fi

# **Command 3: queryer (without --smpl)**
# Only execute if all previous commands have succeeded
if [ "$has_failed" = false ]; then
    echo "--- (3/3) Running the second queryer command... ---"
    CMD3="mpirun -np ${NP_VAL} ./queryer --load ${LOAD_DIR}"
    # Add other arguments
    if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
        CMD3+=" ${OTHER_ARGS[@]}"
    fi

    echo "Executing: $CMD3"
    eval $CMD3
    if [ $? -ne 0 ]; then
        echo "Error: Command 3 failed."
        has_failed=true
    fi
    echo "--- Command 3 finished ---"
    echo
fi

# --- Final Status Report ---
if [ "$has_failed" = true ]; then
    echo "One or more commands failed."
    exit 1
else
    echo "All commands executed successfully."
fi

exit 0