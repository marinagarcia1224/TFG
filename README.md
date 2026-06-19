Analysis of the $H \to \gamma\gamma$ decay with ATLAS Open Data

This repository contains the necessary scripts to perform a simplified analysis of the Higgs boson decay into two photons ($H \to \gamma\gamma$) using the ATLAS Open Data dataset at $\sqrt{s} = 13$ TeV. The analysis is divided into two main stages, represented by the two main scripts:

1. v12.py (Real data analysis). This script processes the real data collected by the ATLAS detector.
  
   -Event selection and categorization. It also documents how many events survive each cut and how they are distributed across the different categories, exporting a summary table.
   
   -Signal + Background Fit. Uses the DSCB function with the tail parameters fixed to the values obtained from the MC. The background is modeled with a polynomial whose degree depends on the available statistics of each category.
   
   -Results extraction. Calculates the fitted Higgs mass, resolution, number of signal events, and the statistical significance of the discovery in each channel, generating plots with the main panel and a residuals panel (background subtraction).

   HOW TO RUN (15min)
   
   Bash
   python v12.py
   
   This generates a folder containing the results.

3. mc10.py (Monte Carlo analysis). This script analyzes the simulated signal samples (ggH, VBF, VH, ttH) and the continuous background ($\gamma\gamma$).
   
   -Shape plots. Generates histograms to compare the kinematic distributions of the different production modes (jet/lepton multiplicity, $m_{jj}$,   $Z_{\gamma\gamma}$, etc.).
   
   -Composition matrices. Calculates and plots efficiency and purity matrices (applies the exact same selection and categorization chain as the v12.py)

   -Signal fits. Performs fits on the Higgs mass peak using functions such as Gaussian, Breit-Wigner, and DSCB. The resulting parameters from the DSCB are saved to be used later on real data.


   HOW TO RUN (1min)
   
   Bash
   python mc10.py
  
   This generates a folder containing the results.

