# Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity


This repository contains the official implementation of "Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity".

## Overview

Dominated Novelty Search (DNS) is a novel Quality-Diversity algorithm that implements local competition through dynamic fitness transformations, eliminating the need for predefined bounds or parameters. Our method:

- Outperforms existing approaches across standard Quality-Diversity benchmarks
- Maintains high performance in high-dimensional and unsupervised spaces
- Serves as a drop-in replacement for grid mechanisms in MAP-Elites

## Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/adaptive-intelligent-robotics/Dominated-Novelty-Search
cd Dominated-Novelty-Search

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
