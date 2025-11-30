#!/bin/bash

# Default to measuring both CPU 0 and 1 if no argument is provided
if [ -z "$1" ]; then
  CPUS_TO_MEASURE="0 1"
  echo "Usage: $0 <CPU_ID>"
  echo "Example: $0 0  (Measure CPU 0)"
  echo "         $0 1  (Measure CPU 1)"
  echo "No CPU ID provided, defaulting to measuring CPUs 0 and 1."
else
  CPUS_TO_MEASURE=$1
fi

for CPU_ID in $CPUS_TO_MEASURE; do
  RAPL_PATH="/sys/class/powercap/intel-rapl:${CPU_ID}"

  # Check if the directory exists
  if [ ! -d "$RAPL_PATH" ]; then
    echo "Error: RAPL directory for CPU ${CPU_ID} not found at ${RAPL_PATH}"
    continue # Skip to the next CPU if the directory doesn't exist
  fi

  echo "Measuring power consumption for CPU ${CPU_ID}..."

  # Read the initial energy value in microjoules
  energy1=$(cat "${RAPL_PATH}/energy_uj")
  # Wait for one second
  sleep 1
  # Read the final energy value in microjoules
  energy2=$(cat "${RAPL_PATH}/energy_uj")

  # Calculate the energy difference
  energy_delta=$((energy2 - energy1))
  # Time difference is 1 second
  time_delta=1

  # Calculate the average power in Watts (microjoules/second = microwatts)
  # then divide by 1,000,000 to get Watts.
  power_watts=$(echo "scale=2; $energy_delta / $time_delta / 1000000" | bc)

  echo "Average power for CPU ${CPU_ID}: $power_watts W"
  echo # Add a blank line for better readability between multiple CPU outputs
done