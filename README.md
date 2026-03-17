# Disease Progression Modeling for Sepsis Using MT-TICC

A self-learning tutorial implementing disease progression modeling for sepsis patients using a simplified version of the **MT-TICC** (Multi-series Time-aware Toeplitz Inverse Covariance-based Clustering) algorithm, applied to clinical data from the MIMIC-III database.

Based on the paper:
> Yang, X., Zhang, Y., & Chi, M. (2021). *Multi-series Time-aware Sequence Partitioning for Disease Progression Modeling.* Department of Computer Science, North Carolina State University.

## What This Tutorial Does

This tutorial walks through a complete pipeline for modeling how sepsis patients progress through distinct clinical states over time, and then uses those discovered states to predict septic shock earlier. The pipeline has five stages:

1. **Data Extraction** - SQL queries to pull sepsis cohorts, vital signs, and lab results from MIMIC-III (or generate synthetic equivalents)
2. **Preprocessing** - Outlier removal, imputation of ~80% missing data, z-score normalization, and cohort balancing
3. **MT-TICC Clustering** - Unsupervised discovery of 6 disease states using sliding-window subsequence clustering with multi-series and time-aware consistency
4. **Cluster Interpretation** - Heatmaps of feature deviations, transition probability matrices (shock vs. non-shock), and patient timeline visualizations
5. **LSTM Prediction** - Bidirectional LSTM comparing Original features (O) vs. Original + Cluster features (O+C) for septic shock early warning

The tutorial produces three key visualizations:

- **Cluster Deviations Heatmap** - Shows how each of the 6 discovered clusters deviates from normal clinical ranges across 14 features
- **Transition Matrices** - Side-by-side comparison of how shock patients and non-shock patients transition between disease states
- **Patient Timelines** - Color-coded disease progression trajectories for individual patients

## Important: Synthetic Data by Default

This tutorial uses **synthetic clinical data** by default (`USE_SYNTHETIC = True`). The synthetic data generator creates 200 ICU patients with 6 discrete clinical states that mirror the paper's discovered clusters (Table II).

**Why not real MIMIC-III data?** Our implementation is a *simplified* version of MT-TICC that omits three critical algorithmic components from the full paper. After extensive testing with real MIMIC-III data, the simplified clustering produced undifferentiated clusters and the LSTM prediction model remained near chance level (AUC ~0.55–0.60). Synthetic data with planted cluster structure allows the tutorial to demonstrate all pipeline concepts effectively. See the [Limitations](#limitations) section below for full details.

You can switch to real MIMIC-III data by setting `USE_SYNTHETIC = False` in the `run_full_pipeline()` function, but be aware that the simplified algorithm is unlikely to produce clinically meaningful results on real data.

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd SelfLearningTutorial
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy psycopg2-binary torch
```

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.21 | Array operations, linear algebra |
| pandas | >= 1.3 | DataFrame manipulation |
| matplotlib | >= 3.4 | Plotting and visualization |
| seaborn | >= 0.11 | Statistical heatmaps |
| scikit-learn | >= 0.24 | GMM clustering, metrics, preprocessing |
| scipy | >= 1.7 | Statistical functions |
| torch (PyTorch) | >= 1.9 | Bidirectional LSTM model |
| psycopg2-binary | >= 2.9 | PostgreSQL connection (only needed for real MIMIC-III) |

PyTorch will auto-detect your hardware - Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

### 3. Run the Tutorial

```bash
.venv/bin/python3 disease_progression_modeling_tutorial.py
```

By default this runs on synthetic data and takes approximately 2–5 minutes. The script will print progress at each pipeline stage and save three PNG visualizations to the working directory.

### 4. Review Outputs

After running, you will see:

- **Console output** with cluster interpretations, transition statistics, and LSTM metrics (accuracy, recall, precision, F1, AUC) for both O and O+C feature sets
- **cluster_deviations.png** - Feature deviation heatmap across 6 clusters
- **transitions.png** - Side-by-side shock vs. non-shock transition matrices
- **timeline.png** - Per-patient disease progression timelines

## Using Real MIMIC-III Data (Optional)

If you have MIMIC-III set up locally on PostgreSQL:

1. **Install MIMIC-III**: Follow the [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) build scripts to create the `mimiciii` schema
2. **Verify your installation**:
   ```bash
   psql -d mimic -c "SELECT count(*) FROM mimiciii.patients"
   # Should return ~46,520
   ```
3. **Configure credentials**: Edit `MIMIC_DB_CONFIG` near the top of the script:
   ```python
   MIMIC_DB_CONFIG = {
       'dbname':   'mimic',
       'user':     'your_username',   # <-- change this
       'password': '',                 # leave empty for local socket auth
       'host':     'localhost',
       'port':     '5432',
   }
   ```
4. **Switch the flag**: In `run_full_pipeline()`, set:
   ```python
   USE_SYNTHETIC = False
   ```
5. **Run**: `.venv/bin/python3 disease_progression_modeling_tutorial.py`

**Note:** The simplified MT-TICC implementation may not produce meaningful clusters on real data. See [Limitations](#limitations).

## The 14 Clinical Features

The tutorial extracts and models 14 clinical features matching the paper:

**Vital Signs (6):** Systolic Blood Pressure (SBP), Mean Arterial Pressure (MAP), Heart Rate (HR), Respiratory Rate (RR), Oxygen Saturation (SpO2), Temperature

**Lab Results (8):** White Blood Cell Count (WBC), Bilirubin, Blood Urea Nitrogen (BUN), Lactate, Creatinine, Platelet Count, Neutrophils, Fraction of Inspired Oxygen (FiO2)

## The 6 Discovered Disease States

The clustering discovers 6 states that map to clinically interpretable conditions (matching the paper's Table II):

| Cluster | Clinical Interpretation | Key Features |
|---------|----------------------|--------------|
| C1 | Near-normal / mild metabolic | Vitals and labs close to normal ranges |
| C2 | Renal dysfunction | Elevated BUN and creatinine |
| C3 | Physiological stress response | Elevated HR, WBC, mild hypotension |
| C4 | Metabolic / hepatic | Elevated bilirubin, temperature, FiO2 |
| C5 | Severe multi-organ dysfunction | High lactate, low platelets, hypotension |
| C6 | Organ failure / critical | Widespread severe abnormalities across all systems |

## Key Concepts from the Paper

### TICC (Toeplitz Inverse Covariance-based Clustering)

TICC clusters multivariate time-series subsequences by learning a sparse inverse covariance (precision) matrix for each cluster. The precision matrix captures *conditional* dependencies between features - which variables directly influence each other after accounting for everything else. The block Toeplitz constraint enforces that temporal relationships at the same lag are identical regardless of position in the window.

### Multi-Series Extension (M-TICC)

Real EHR data contains multiple patient visits. Rather than treating each visit independently (inconsistent patterns) or concatenating them (artificial boundary effects), M-TICC jointly clusters all series to learn shared structural patterns across patients.

### Time-Awareness Extension (MT-TICC)

EHR measurements are irregularly sampled - intervals range from seconds to 28+ hours. MT-TICC introduces a time-aware consistency term:

```
c(X_{t-1}, P_k, Δt) = β · 1{t-1 ∉ P_k} / log(e + Δt)
```

This penalizes cluster switches *inversely proportional* to the time gap: close-in-time events are strongly encouraged to share a cluster, while events far apart are allowed to transition naturally.

### Hold-off Window (τ)

For prediction, the tutorial uses a hold-off window τ: given observations up to τ hours *before* the endpoint (shock onset or discharge), predict shock vs. non-shock. The paper shows MT-TICC provides the largest improvement over baselines at larger τ (earlier, harder predictions), because cluster features encode structural patterns that remain informative even when recent clinical values are unavailable.

## Limitations

This tutorial implements a **simplified** MT-TICC. Three key algorithmic components from the full paper are omitted, which is why synthetic data is used by default:

### 1. No Graphical Lasso

The full MT-TICC learns **sparse precision matrices** per cluster using ADMM (Alternating Direction Method of Multipliers). The precision matrix captures *conditional* dependencies - which features directly influence each other. Our simplified version uses standard covariance matrices from Gaussian Mixture Models, which capture *marginal* correlations. Marginal correlations are dominated by confounders (e.g., "sicker patients have worse values on everything"), so clusters collapse into near-identical profiles on real data.

### 2. No Viterbi Dynamic Programming

The full algorithm uses a Viterbi-like dynamic programming E-step to find the **globally optimal** cluster assignment sequence for each patient's entire stay. Our simplified version uses a greedy forward pass that picks the locally best cluster at each time point. This myopic approach can get stuck in suboptimal assignments that cascade through the rest of the sequence.

### 3. No Block Toeplitz Constraint

The full MT-TICC constrains precision matrices to have a block Toeplitz structure, enforcing that temporal relationships at the same lag are **position-invariant**. Without this constraint, the model has more parameters to estimate and less inductive bias for learning generalizable time-series patterns.

### What This Means in Practice

Without these three components, our implementation is essentially **K-Means on windowed subsequences with a temporal consistency penalty**. This is sufficient for demonstrating all pipeline concepts on synthetic data with planted cluster structure, but insufficient for discovering meaningful clusters from raw real-world EHR data or reproducing the paper's quantitative results.

For the full algorithm, see the [TICC library by Hallac et al.](https://github.com/davidhallac/TICC) as a starting point.

## Repository Structure

```
SelfLearningTutorial/
├── README.md                                          # This file
├── disease_progression_modeling_tutorial.py            # Main tutorial script (~2500 lines)
├── Disease_Progression_Modeling_Tutorial.pptx          # Companion presentation (18 slides)
├── Multi-series Time-aware Sequence Partitioning...pdf # Reference paper
├── cluster_deviations.png                             # Generated: feature deviation heatmap
├── transitions.png                                    # Generated: shock vs non-shock transitions
└── timeline.png                                       # Generated: patient progression timelines
```

## Code Structure

The tutorial script is organized into 11 sections:

| Section | Description |
|---------|-------------|
| 1 | Environment setup and imports |
| 2 | MIMIC-III data extraction (SQL queries for cohort, vitals, labs) |
| 3 | Data preprocessing (outlier removal, imputation, normalization) |
| 4 | Understanding TICC - conceptual foundation |
| 5 | Simplified MT-TICC implementation (GMM init, greedy E-step, covariance M-step) |
| 6 | Cluster interpretation and analysis |
| 7 | Bidirectional LSTM for septic shock prediction (PyTorch) |
| 8 | Visualization functions (heatmaps, transitions, timelines) |
| 9 | Main pipeline orchestration |
| 10 | Synthetic data generation |
| 11 | Limitations and discussion |

Every function includes detailed docstrings and inline comments explaining the clinical and algorithmic rationale behind each decision.

## References

1. Yang, X., Zhang, Y., & Chi, M. (2021). Multi-series Time-aware Sequence Partitioning for Disease Progression Modeling. NC State University.
2. Hallac, D., Vare, S., Boyd, S., & Leskovec, J. (2017). Toeplitz Inverse Covariance-based Clustering of Multivariate Time Series Data. SIGKDD.
3. Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA, 315.
4. Johnson, A.E.W., et al. (2016). MIMIC-III, a Freely Accessible Critical Care Database. Scientific Data.
5. Lipton, Z.C., Kale, D.C., & Wetzel, R. (2015). Learning to Diagnose with LSTM Recurrent Neural Networks. arXiv:1511.03677.
6. Baytas, I.M., et al. (2017). Patient Subtyping via Time-Aware LSTM Networks. SIGKDD.

## Author

**Jiel Selmani** - March 2026

## License

This tutorial is for educational purposes. MIMIC-III data access requires completion of the [PhysioNet credentialing process](https://physionet.org/content/mimiciii/).
