# 🏥 An Explainable Clinical Decision Support System for Similarity-Based Treatment Recommendation in ICU Patients

---

## 📌 Introduction

Intensive Care Units (ICUs) generate high-dimensional clinical data that must be interpreted rapidly to guide treatment decisions. In real-world practice, clinicians often rely on prior experience with similar patients when selecting interventions. However, systematically identifying comparable cases from large-scale historical datasets is challenging.

This project presents an **explainable Clinical Decision Support System (CDSS)** that leverages structured ICU data to:

- Identify clinically similar patient cohorts  
- Recommend treatment categories based on outcomes and interventions observed in comparable cases  

The system is designed to **support, not replace, clinical judgment**.

---

## 🗂 Dataset

The system is developed using de-identified critical care data from the **MIMIC-IV** database available via **PhysioNet**.

MIMIC-IV is a modern, relational ICU dataset widely used in clinical research.

The project utilizes:

- Patient demographics  
- Hospital admissions  
- ICU stay records  
- Vital signs  
- Laboratory measurements  
- Diagnoses  
- Treatment events  

To maintain focus and reduce complexity:

- Emergency department data is excluded  
- Free-text clinical notes are excluded  

---

## ❓ Problem Statement

ICU datasets contain rich longitudinal information, yet clinicians lack tools that can:

- Efficiently identify historically similar patients  
- Summarize treatments effective in comparable contexts  
- Provide interpretable recommendations  

There is a clear need for **transparent, ethically constrained systems** that operationalize patient similarity without compromising safety or interpretability.

---

## 🎯 Objectives

- Construct a patient representation capturing physiological state, comorbidities, and early ICU measurements  
- Identify clinically similar patients using similarity or distance-based methods  
- Recommend relevant treatment categories based on comparable patient cohorts  
- Ensure transparency and interpretability of recommendations  
- Develop an ethically constrained clinical decision support system  

---

## ⚙️ Methodology

### 1️Data Processing

Structured data from selected MIMIC-IV tables are:

- Preprocessed  
- Aggregated over clinically meaningful windows (e.g., first 24 hours of ICU admission)  
- Engineered to avoid data leakage  

### 2️Feature Engineering

Features include:

- Demographics  
- Aggregated vital signs  
- Laboratory extremes  
- Comorbidity indicators  
- Treatment usage flags  

### 3️Patient Similarity Modeling

Similarity is computed using:

- Cosine similarity  
- Euclidean distance  
- Embedding-based representations  

### 4️Treatment Recommendation

Treatment recommendations are derived from patterns observed among the most similar patient cohorts.

### 5️Explainability

Explainability is achieved through:

- Feature contribution analysis  
- Case-based similarity comparison  
- Transparent ranking of similar historical patients  

---

## 📈 Expected Outcomes

The system outputs:

- A ranked set of clinically similar historical patients  
- Recommended treatment categories derived from those cohorts  
- Feature-level explanations highlighting influential patient attributes  

The system functions strictly as a **decision support tool**, with final treatment decisions remaining under clinician control.

---

## ⚖️ Ethical Considerations

- All data are de-identified and comply with PhysioNet data usage requirements  
- The system does not perform diagnosis or autonomous treatment selection  
- Emphasis is placed on transparency, auditability, and responsible use of historical clinical data  

---

## 🏁 Conclusion

This project demonstrates the feasibility of a similarity-based, explainable CDSS for ICU treatment support using modern clinical data. By grounding recommendations in comparable patient cases and emphasizing interpretability, the system aligns with real-world clinical reasoning while maintaining ethical and safety standards.
