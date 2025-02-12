#!/bin/bash

# Set the split size
split_size="90M"

# Split the model.bin file
split -b "$split_size" model.bin part_model_

# Split the model_quant.bin file
split -b "$split_size" model_quant.bin part_quant_

echo "File splitting completed."