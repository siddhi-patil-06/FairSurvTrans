# 🧠 FairSurvTrans: Fairness-Aware Transformer for Survival Prediction

FairSurvTrans is a deep learning survival analysis model that combines transformer architecture with fairness-aware optimization. It is designed to predict 1-year survival outcomes for patients undergoing hematopoietic cell transplantation (HCT), while minimizing racial disparities in prediction performance.

---

## 📊 Dataset

- **Source**: Kaggle FairSurvival Challenge (synthetic data)
- **Samples**: 24,493 synthetic patient records
- **Target**: `efs_time` (event-free survival time), `efs` (event indicator)
- **Balanced racial groups**:
  - White
  - Asian
  - African-American
  - Native American
  - Pacific Islander
  - Multiracial

---

## 🧠 Model Architecture

FairSurvTrans adapts Transformer architecture for structured survival data.

- **Mixed Input Embedding**:
  - Categorical → Learned embeddings
  - Numerical → Linear projections

- **Transformer Encoder**:
  - Multi-head self-attention
  - Feed-forward network with GELU
  - Residual connections + LayerNorm

- **Feature Attention Pooling**:
  \[
  h = \sum_i \alpha_i X_i,\quad \alpha = \text{softmax}(W_2 \cdot \tanh(W_1 X))
  \]

- **Risk Prediction Head**:
  \[
  \hat{y} = W_{\text{out}} \cdot \text{GELU}(W_h h + b_h) + b_{\text{out}}
  \]

---

## ⚖️ Fairness-Aware Loss Function

The model minimizes both prediction error and racial disparities.

**Total Loss**:
\[
L_{\text{fair}} = L_{\text{Cox}} + \lambda \cdot \frac{1}{R} \sum_{r=1}^{R} (C_{\text{overall}} - C_r)^2
\]

- \(L_{\text{Cox}}\): Cox partial log-likelihood
- \(C_r\): C-index for racial group \(r\)
- \(C_{\text{overall}}\): Overall concordance index
- \(\lambda\): Fairness penalty weight

---

## 🔁 Training & Optimization

- **Preprocessing**:
  - Missing values → KNNImputer
  - Categorical encoding → OrdinalEncoder
  - Normalization → StandardScaler

- **Cross-validation**:
  - 5-Fold Stratified by race

- **Dynamic Subgroup Weighting**:
  \[
  w_r = \text{clip}(1 + \alpha (C_{\text{avg}} - C_r), 0.5, 2.0)
  \]

- **Optimizer**: Adam
- **Regularization**: Dropout, Early stopping

---

## 📈 Evaluation Metrics

| Metric              | Description                              |
|---------------------|------------------------------------------|
| C-index             | Concordance index for ranking accuracy   |
| Stratified C-index  | Per-race group performance               |
| Fairness Gap        | Std. dev. of subgroup C-indices          |

Example results:
- **C-index**: ~0.674  
- **Fairness Gap**: 0.0002 (lowest among baselines)

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook fairsurvtrans-final-code.ipynb
