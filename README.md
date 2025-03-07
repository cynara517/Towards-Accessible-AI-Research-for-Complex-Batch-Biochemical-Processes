<h1 align="center">
Enabling Reproducible Research in Batch Chemical Processes  
A Privacy-Preserving Preprocessing Pipeline, Benchmark Datasets, and Spatiotemporal Kinetic Modeling
</h1>

<p align="center">
  <img alt="GitHub" src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

<style>
  .section-blue { background-color: #E3F2FD; padding: 15px; border-radius: 10px; }
  .section-green { background-color: #E8F5E9; padding: 15px; border-radius: 10px; }
  .section-yellow { background-color: #FFF9C4; padding: 15px; border-radius: 10px; }
  .section-red { background-color: #FFEBEE; padding: 15px; border-radius: 10px; }
</style>

<div class="section-blue">
  <h2>🌟 Key Contributions</h2>
  <p><b>🔒 Three Major Contributions</b></p>
  <ul>
    <li><b>Publicly Available Industrial Batch Data</b>: Dataset publication based on process dynamics temporal segmentation strategies and privacy-preserving techniques, along with mechanistic relationship diagrams.</li>
    <li><b>PPP (Proposed Preprocessing Pipeline)</b>: A standardized preprocessing framework tailored to the unique characteristics of batch chemical processes.</li>
    <li><b>PPPP (Proposed Privacy-Preserving Pipeline)</b>: A dataset generation pipeline incorporating advanced privacy protection strategies.</li>
  </ul>
</div>

---

<div class="section-green">
  <h2>🔬 Dataset Overview</h2>
  <h3>🔒 Dimensional Definitions</h3>
  <ul>
    <li><b>B</b>: Number of Batches</li>
    <li><b>T</b>: Time Steps per Batch</li>
    <li><b>F</b>: Feature Dimensions</li>
  </ul>
  
  <h3>📊 Benchmark Dataset</h3>
  <table align="center">
    <tr>
      <th>Dataset</th>
      <th>Dimensions (B×T×F)</th>
      <th>Industry</th>
      <th>Production Characteristics</th>
    </tr>
    <tr>
      <td><b>SF</b></td>
      <td align="center">2×88390×14</td>
      <td>Energy & Power</td>
      <td>Dynamic monitoring of steam flow in thermal power plant boilers, including pressure-temperature coupling control.</td>
    </tr>
    <tr>
      <td><b>UF</b></td>
      <td align="center">3×6979×17</td>
      <td>Food Processing</td>
      <td>Multi-stage process data from ultra-processed food production lines (mixing-sterilization-packaging).</td>
    </tr>
    <tr>
      <td><b>EP</b></td>
      <td align="center">270×100×15</td>
      <td>Pharmaceuticals</td>
      <td>Dissolved oxygen-pH-feeding coordination process in erythromycin fermentation tanks.</td>
    </tr>
  </table>
</div>

---

<div class="section-yellow">
  <h2>💼 Proposed Preprocessing Pipeline (PPP)</h2>
  <h3>🔒 Core Innovations</h3>
  <ul>
    <li><b>Intelligent Batch Segmentation</b>: Fuzzy clustering enables precise segmentation of complex batch processes.</li>
    <li><b>Advanced Data Imputation</b>: Utilizes long-short term sequence dependencies to handle missing values.</li>
    <li><b>Task-Specific Feature Selection</b>: Constructs adaptive variable selection methods tailored to deep learning tasks.</li>
    <li><b>Batch-Aware Normalization</b>: Introduces a specialized normalization approach considering batch length variations.</li>
  </ul>
</div>

---

<div class="section-red">
  <h2>💼 Proposed Privacy-Preserving Pipeline (PPPP)</h2>
  <h3>🔒 Core Innovations</h3>
  <ul>
    <li><b>Bidirectional Temporal Modeling</b>: Captures device operational state evolution via forward-backward dependencies.</li>
    <li><b>Domain Knowledge Integration</b>: Embeds physical-chemical constraints in a differentiable format within the loss function.</li>
    <li><b>Optimized Privacy-Utility Tradeoff</b>: Implements state-of-the-art privacy preservation strategies while maintaining data usability.</li>
  </ul>
  
  <h3>🔧 Usage Instructions</h3>
  <pre><code>
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
  </code></pre>
</div>

---

## 🚀 **Quick Deployment**
```bash
# Install dependencies (recommended: conda virtual environment)
conda create -n bcpenv python=3.9
conda activate bcpenv
pip install -r requirements.txt
