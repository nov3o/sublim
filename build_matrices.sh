#!/bin/bash
# Build matrices for colors, trees, and morality experiments

set -e

echo "========================================="
echo "BUILDING EVALUATION MATRICES"
echo "========================================="

# Colors matrix
echo ""
echo "Building COLOR matrix..."
python3 build_animal_matrix.py \
    --names blue,green,red,purple \
    --eval-type color \
    --output matrix_colors

# Colors semantic matrix
echo ""
echo "Building COLOR (semantic) matrix..."
python3 build_animal_matrix.py \
    --names blue,green,red,purple \
    --modifier semantic \
    --eval-type color \
    --output matrix_colors_semantic

# Trees matrix
echo ""
echo "Building TREE matrix..."
python3 build_animal_matrix.py \
    --names acacia,bamboo,sequoia \
    --eval-type tree \
    --output matrix_trees

# Trees semantic matrix
echo ""
echo "Building TREE (semantic) matrix..."
python3 build_animal_matrix.py \
    --names acacia,bamboo,sequoia \
    --modifier semantic \
    --eval-type tree \
    --output matrix_trees_semantic

# Morality matrix (semantic only)
echo ""
echo "Building MORAL QUALITY (semantic) matrix..."
python3 build_animal_matrix.py \
    --names good,evil \
    --modifier semantic \
    --eval-type moral_quality \
    --output matrix_morality_semantic

echo ""
echo "========================================="
echo "ALL MATRICES BUILT!"
echo "========================================="
echo "Output files:"
echo "  - matrix_colors.json"
echo "  - matrix_colors_semantic.json"
echo "  - matrix_trees.json"
echo "  - matrix_trees_semantic.json"
echo "  - matrix_morality_semantic.json"
