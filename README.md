# Predictive_Looking

Welcome to our GitHub repository where you can find all the necessary code and data for conducting the analysis associated with the predictive looking and predictive looking error project.

## Repository Contents

- **Code/**: This directory contains all the scripts used for analysis, including data processing and statistical testing.
- **Data/**: All datasets used in the analyses are stored here. Please see the data dictionary for detailed descriptions of each file.

## Reproducing Analysis

To replicate the figures in the manuscripts, simply download the repository and run `predictive_looking_analysis.Rmd` and `derive_segmentation_probabilities_publication.Rmd`. If you want to try deriving grain-based gaze density from your own eye-tracking dataset in movie watching, you can refer to `gaze2grid.py`. For reproducing gaze density in each grid for my dataset, please refer to the code and data stored in the OSF directory [insert link here].

### Setting Up Your Environment

To properly utilize `gaze2grid.py` or `hand2grid.py`, you must first ensure that `utils.py` is correctly integrated. Here's how you can set up your scripts:

1. **Add the Import Statement at the Top of Your Script:**
   ```python
   # Import all functions from utils.py
   from utils import *

## Support

If you have any questions, please contact Sophie Su at sophiesu1996@gmail.com