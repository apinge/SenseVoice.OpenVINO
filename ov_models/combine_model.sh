#!/bin/bash

cat part_model_* > model.bin
cat part_quant_* > model_quant.bin

# Calculate the MD5 checksum of model.bin
md5sum model.bin > model.bin.md5

expected_md5="b20530d9e4fc498d7b8642a6d3224784"

# Read the calculated MD5 checksum from the model.bin.md5 file
calculated_md5=$(cat model.bin.md5 | awk '{print $1}')

# Compare the calculated MD5 checksum with the expected MD5 checksum
if [[ "$calculated_md5" == "$expected_md5" ]]; then
  echo "The MD5 checksum of combined model.bin matches the expected value."
else
  echo "Warning: The MD5 checksum of combined model.bin does not match the expected value."
  echo "Expected value: $expected_md5"
  echo "Calculated value: $calculated_md5"
fi

# Remove the temporary MD5 file
rm model.bin.md5


# Calculate the MD5 checksum of model.bin
md5sum model_quant.bin > model_quant.bin.md5

expected_quant_md5="6cc69c945a45e14aba715ae980cb645b"

# Read the calculated MD5 checksum from the model.bin.md5 file
calculated_md5=$(cat model_quant.bin.md5 | awk '{print $1}')

# Compare the calculated MD5 checksum with the expected MD5 checksum
if [[ "$calculated_md5" == "$expected_quant_md5" ]]; then
  echo "The MD5 checksum of combined model_quant.bin matches the expected value."
else
  echo "Warning: The MD5 checksum of combined model_quant.bin does not match the expected value."
  echo "Expected value: $expected_quant_md5"
  echo "Calculated value: $calculated_md5"
fi

# Remove the temporary MD5 file
rm model_quant.bin.md5


echo "File merging and MD5 checksum verification completed."

