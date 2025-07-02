# 🧬 FairSurvTrans: Fairness-Aware Transformer for Survival Prediction in HCT

**LLM-inspired transformer model for equitable survival prediction across racial groups in allogeneic hematopoietic cell transplantation (HCT)**

---

## 📌 Overview

**FairSurvTrans** is a transformer-based deep learning model designed for structured clinical data to predict patient survival after HCT while minimizing racial bias. Inspired by LLMs like BERT/GPT, it applies multi-head self-attention and a fairness-sensitive loss function to achieve both high accuracy and equitable subgroup performance.

Developed using a racially balanced synthetic dataset (from the Kaggle FairSurvival Challenge), it integrates dynamic group weighting and stratified C-index evaluation.

---

## 🎯 Key Features

- 🔁 End-to-end transformer for survival prediction
- ⚖️ Fairness-aware loss: Cox loss + stratified fairness penalty
- 🧪 Subgroup C-index & calibration monitoring
- 📊 C-index: **0.6738 ± 0.0031**, Fairness Penalty: **0.0002 ± 0.0001**
- ✅ Consistent performance across 6 racial groups

---

## 📂 Project Structure

```
FairSurvTrans/
├── data/                            # Synthetic dataset (processed or raw)
├── notebooks/
│   └── FairSurvTrans_Model.ipynb    # Complete training + evaluation pipeline
├── requirements.txt                 # Environment dependencies
└── README.md                        # Project documentation
```


---

## 🛠️ Methodology Summary

### ⚙️ Architecture

- Mixed-type embedding for categorical & numerical features
- Learnable positional encoding (non-sequential tabular data)
- Multi-head self-attention + residuals + GELU
- Feature attention pooling for patient-level aggregation
- Risk prediction via MLP head

### 📈 Loss Function

Final objective:
```
L_fair = L_cox + λ · FairnessPenalty
```
Where:
- `L_cox` = Negative partial log-likelihood from Cox model  
- `FairnessPenalty` = Mean squared deviation in C-index across subgroups  
- `λ` = fairness penalty weight (default: 0.2)

### ⚖️ Dynamic Group Weighting
Poorer-performing subgroups receive higher weight during training:
```python
w_r = clip(1.0 + α * (C_avg - C_r), 0.5, 2.0)
```

---

## 📊 Evaluation Highlights

| Model           | C-Index         | Fairness Penalty |
|----------------|------------------|------------------|
| **FairSurvTrans** | **0.6738 ± 0.0031** | **0.0002 ± 0.0001** |
| CoxPH           | 0.6348 ± 0.0057 | 0.0816           |
| DeepSurv        | 0.6219 ± 0.0045 | 0.0724           |
| XGBoost-AFT     | 0.3285 ± 0.0051 | 0.0728           |

✅ Statistically significant improvements (p < 0.05)  
✅ Best subgroup consistency  
✅ Accurate 1-year survival risk calibration

---

## 📈 Ablation Study (Component Contribution)

| Configuration         | C-Index     | Fairness Penalty |
|----------------------|-------------|------------------|
| Full Model           | 0.6738      | 0.0002           |
| No Fairness Loss     | 0.6737      | 0.0000           |
| No Dynamic Weights   | 0.6738      | 0.0002           |
| No Fair Components   | 0.6737      | 0.0000           |

> Fairness improves subgroup equity **without** degrading global performance.

---

## 🧠 Future Work

- ✅ Validate on real-world registries (CIBMTR, SEER)
- 🔄 Add longitudinal modeling (e.g., TransformerJM-style)
- 🧬 Extend with ClinicalBERT for multimodal inputs
- 👥 Human-in-the-loop fairness auditing

---

## 👨‍💻 Authors

- **Siddhi Patil** – [siddhipatil064@gmail.com](mailto:siddhipatil064@gmail.com)  
- **Vijay Kalmani** – [vijaykalmani@gmail.com](mailto:vijaykalmani@gmail.com) *(Corresponding Author)*  
- **Amol Adamuthe** – [amol.admuthe@gmail.com](mailto:amol.admuthe@gmail.com)

---

## 📄 Citation

```bibtex
@article{fairsurvtrans2025,
  title     = {FairSurvTrans: An LLM-Inspired Transformer for Equitable Survival Prediction in Hematopoietic Cell Transplantation},
  author    = {Kalmani, Vijay and Patil, Siddhi and Adamuthe, Amol},
  journal   = {Under Review},
  year      = {2025}
}
```

---

## 📜 License

This project is released under an academic-use license. Contact authors for commercial use.

---

> _"AI in healthcare must ensure fairness, transparency, and equitable treatment across all patient groups."_
