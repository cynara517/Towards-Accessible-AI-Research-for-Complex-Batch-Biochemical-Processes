# <h1 align="center">

<div align="center">
  <h2>Enabling Reproducible Research in Batch Chemical Processes：A Privacy-Preserving Preprocessing Pipeline, Benchmark Datasets, and Spatiotemporal Kinetic Modeling</h2>
  <p>Integrating Privacy-Preserving Data Preprocessing, Standardized Benchmark Datasets, and Domain Knowledge Fusion</p>
</div>
<p align="center">
  <img alt="GitHub" src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## 🌟 Key Contributions

### 🔒 Three Major Contributions
- **Publicly Available Industrial Batch Data**: Dataset publication based on process dynamics temporal segmentation strategies and privacy-preserving techniques, along with mechanistic relationship diagrams.
- **PPP (Proposed Preprocessing Pipeline)**: A standardized preprocessing framework tailored to the unique characteristics of batch chemical processes.
- **PPPP (Proposed Privacy-Preserving Pipeline)**: A dataset generation pipeline incorporating advanced privacy protection strategies.

---

## 🔬 Dataset Overview
### **🔒Dimensional Definitions**
- **B**: Number of Batches
- **T**: Time Steps per Batch
- **F**: Feature Dimensions

### **📊Benchmark Dataset**
<table align="center">
  <tr>
    <th>Dataset</th>
    <th>Dimensions (B×T×F)</th>
    <th>Industry</th>
    <th>Production Characteristics</th>
  </tr>
  <tr>
    <td><b>SF</b><br></td>
    <td align="center">2×88390×14</td>
    <td>Energy & Power</td>
    <td>Dynamic monitoring of steam flow in thermal power plant boilers, including pressure-temperature coupling control.</td>
  </tr>
  <tr>
    <td><b>UF</b><br></td>
    <td align="center">3×6979×17</td>
    <td>Food Processing</td>
    <td>Multi-stage process data from ultra-processed food production lines (mixing-sterilization-packaging).</td>
  </tr>
  <tr>
    <td><b>EP</b><br></td>
    <td align="center">270×100×15</td>
    <td>Pharmaceuticals</td>
    <td>Dissolved oxygen-pH-feeding coordination process in erythromycin fermentation tanks.</td>
  </tr>
  <tr>
    <td><b>TE</b><br></td>
    <td align="center">1×1498×33</td>
    <td>Chemical Nitration</td>
    <td>Monitoring of temperature gradients and product concentration in nitration reactors.</td>
  </tr>
  <tr>
    <td colspan="4" align="center"><i>Comparative Study of Penicillin Production Control Modes</i></td>
  </tr>
  <tr>
    <td>PPR</td>
    <td align="center">30×895×11</td>
    <td rowspan="3">Pharmaceuticals</td>
    <td>Formula-driven mode (fixed parameter curves)</td>
  </tr>
  <tr>
    <td>PPO</td>
    <td align="center">30×965×11</td>
    <td>Manual control mode (operator expertise adjustments)</td>
  </tr>
  <tr>
    <td>PPAPC</td>
    <td align="center">30×835×11</td>
    <td>Advanced process control (real-time Raman spectroscopy feedback)</td>
  </tr>
</table>
---

##  💼 **Proposed Preprocessing Pipeline (PPP)**
### **🔒Core Innovations**
- **Intelligent Batch Segmentation**: Fuzzy clustering enables precise segmentation of complex batch processes.
- **Advanced Data Imputation**: Utilizes long-short term sequence dependencies to handle missing values.
- **Task-Specific Feature Selection**: Constructs adaptive variable selection methods tailored to deep learning tasks.
- **Batch-Aware Normalization**: Introduces a specialized normalization approach considering batch length variations.

### **🔧Workflow**
mermaid
flowchart TD;
    A[Raw Time-Series Data] -->|SCFCM Clustering| B;
    B -->|Missing Data Imputation| C;
    C -->|MIC Pairwise| C1;
    C -->|MIC Target| C2;
    C1 -->|Normalization| D[Structured Data];
    C2 -->|Normalization| D;



---

##  💼**Proposed Privacy-Preserving Pipeline (PPPP)**

### **🔒Core Innovations**
- **Bidirectional Temporal Modeling**: Captures device operational state evolution via forward-backward dependencies.
- **Domain Knowledge Integration**: Embeds physical-chemical constraints in a differentiable format within the loss function.
- **Optimized Privacy-Utility Tradeoff**: Implements state-of-the-art privacy preservation strategies while maintaining data usability.


### **🔧Usage Instructions**
bash

opt.py  # Configure core parameters

GANBILSTM.py  # Execute training process

python GANBILSTM.py --mode train

trainer.py  # Manages training execution within GANBILSTM.py

metrics.py  # Automatically computes relevant performance metrics

loader.py  # Loads datasets using get_loader() within GANBILSTM.py and trainer.py

# The model automatically saves the best parameters as best_model.pth
# Use resume_path to reload parameters and run prediction mode

python GANBILSTM.py --mode predict

# Model results saved in dataname_test_evaluate.txt
# Predictions stored in predictions.csv


## 🚀 **Quick Deployment**
bash
# Install dependencies (recommended: conda virtual environment)
conda create -n bcpenv python=3.9
conda activate bcpenv
pip install -r requirements.txt

##
The official citation format for this framework and dataset will be updated upon the paper's formal publication.
