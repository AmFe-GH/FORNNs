# Repository :rocket: 
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) 
![version](https://img.shields.io/badge/version-v2.1.0-blue) 
![Dependency](https://img.shields.io/badge/dependency-PyTorch-orange)
![Language](https://img.shields.io/badge/language-Python-blue)
![Contributors](https://img.shields.io/badge/contributors-3-p)
<p align="center">
<img src="/Manuscript/Figure/workflow.png"/> 
</p>
This repository contains the code to replicate the results presented in the paper "Caputo fractional order recurrent neural networks: efficiently modeling dynamic systems with state spaces". 

## Abstract
  Recurrent neural networks (RNNs) endowed with continuous-time 
  states have emerged as an adaptive framework for modeling dynamic
  systems. Among these systems, Caputo fractional order 
  ordinary differential equation  systems (CFODEs) are gaining prominence
  due to their non-local characteristics over time. 
  This study theoretically demonstrates, for the first time,
  the capability of fractional-order RNNs (FORNNs) to universally approximate
  CFODEs  with arbitrary precision.
  Concurrently, during the application of FORNNs to practical scenarios,
  the negative impact of the complexity of parameter space and
  ABM solver on FORNNs' performance is first revealed,
  which are referred to as "Parameter Domain Problems"(PDPs).
  In response to PDPs,
  We propose
  the Local Domain Normalization (LDN),
  along with introducing a novel loss function to rectify the
  Hallucination Problem encountered during the learning process.
  Finally, two real-world examples are presented and validate
  the superior performance of FORNNs,
  which are entirely consistent with the theoretical proofs,
  and also confirm the effectiveness of  LDN method.

## Files

- `main.py`: This is the main file. Running this file can fully demonstrate the generation and processing of training data, as well as the learning process and results of FORNNs.
- `Hyperparameters.py`: Contains preset parameter information used in the experiments.
- `LDN.py`: Implements the Local Domain Normalization method described in the paper.
- `model.py`: Includes model architectures and loss functions used in the experiments.
- `utils.py`: Contains utility functions used in the main file.

## Instructions

To replicate the results:

1. Clone this repository to your local machine:  
   `git clone https://github.com/AmFe-GH/FORNNs`
2. Enter your conda environment:   
   `conda activate <your_env_name>`
3. Install the required packages:  
   `pip install -r requirements.txt`
4. Run `main.py`.  


## Citation

If you use this code in your research or find it helpful, please consider citing the original paper:

[not published yet]



<!-- ## <img src="https://joonsungpark.s3.amazonaws.com:443/static/assets/characters/profile/Wolfgang_Schulz.png" alt="Generative Wolfgang">   Acknowledgements

This work was supported by National Natural Science Foundation of China
(Grant 12101430) and Department of Science and Technology of Sichuan
Province (Grant 2021ZYD0018). (Corresponding author: Cong Wu.)(https://yjs.cd120.com/contents/559/1710.html) -->
