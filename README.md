<div align="center">

# 🌊 H2O Machine Learning Assignment

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![H2O.ai](https://img.shields.io/badge/H2O.ai-ML%20Framework-FFD700?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Iris%20UCI-00C49A?style=for-the-badge)

*Multi-model ML pipeline using H2O.ai — Random Forest, Neural Networks, and Tuned Deep Learning.*

</div>

---

## 🏗️ Pipeline

<div align="center">
  <img src="pipeline_diagram.png" width="860"/>
</div>

```mermaid
flowchart LR
    A["🌸 Iris CSV"] --> B["⚙️ H2O Cluster"]
    B --> C["�� Preprocess\n·Impute ·asFactor ·80/20 Split"]
    C --> D["🌲 Random Forest\nLogloss: 0.1009"]
    C --> E["🧠 Simple Neural Net\nLogloss: 0.1839"]
    C --> F["⚡ Tuned Deep Learning\nLogloss: 0.0915 ⭐"]
```

---

## 📊 Results

<div align="center">
  <img src="model_comparison.png" width="640"/>
</div>

| Rank | Model | Logloss |
|------|-------|---------|
| 🥇 | Tuned Deep Learning `[50,50]` · 50 epochs | **0.0915** |
| 🥈 | Random Forest — 50 trees | 0.1009 |
| 🥉 | Simple Neural Net — hidden=[10] | 0.1839 |

---

## 📂 Files

| File | Description |
|------|-------------|
| `main.py` | Full H2O pipeline — train, grid search, evaluate |
| `Report.md` | Academic report with math & methodology |
| `iris.csv` | UCI Iris dataset |
| `H2O_Assignment_Submission.zip` | Submission archive |

---

## 🚀 Run

```bash
pip install h2o pandas
python main.py
```

<div align="center">
<sub>© 2026 Vivek Raj Singh · H2O.ai · Python · UCI ML Repository</sub>
</div>
