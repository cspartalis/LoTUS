#!/bin/bash

# CIFAKE dataset path
cifake_path="$HOME/data/cifake/test/FAKE"

# Make a new dir named cifake_classes if not exists
mkdir -p "$HOME/data/cifake_classes"

# Loop over class labels 0-9
for i in {0..9}; do
    class_dir="$HOME/data/cifake_classes/$((i))"
    mkdir -p "$class_dir"

    # Find and move files containing (i-1)
    find "$cifake_path" -type f -name "*($((i+1)))*" -exec mv {} "$class_dir/" \;
done

# In the end copy the remaining images (airplanes) to subfolder 0