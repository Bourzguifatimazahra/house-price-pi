# House Price Prediction Interval Challenge
### 3‑Day Individual Assignment — Data Engineering • ML/AI • DevOps • Analytics

---

## 🎯 Challenge Summary

In this assignment, you work with the dataset from  **Prediction Interval Competition II — House Price**.  
Your goal is to build an **objective**, **data‑driven**, **probabilistic forecasting pipeline** that predicts **prediction intervals** (lower and upper bounds) for house prices in King County.

You receive three datasets:

- **dataset.csv** — the full historical training dataset containing sale prices, property attributes, geographic information and environmental indicators [1](https://sachaartemit-my.sharepoint.com/personal/sacha_artemit_nl/_layouts/15/Doc.aspx?sourcedoc=%7B82762BE0-192F-4F9B-8B4B-A99CA0EC9221%7D&file=dataset.csv&action=default&mobileredirect=true)  
- **test.csv** — property feature rows without sale prices; your model must generate prediction intervals for these records [2](https://sachaartemit-my.sharepoint.com/personal/sacha_artemit_nl/_layouts/15/Doc.aspx?sourcedoc=%7B71B456C2-DC28-41E1-9A91-D53C05EF0B0B%7D&file=test.csv&action=default&mobileredirect=true)  
- **sample_submission.csv** — the required output structure (`id`, `pi_lower`, `pi_upper`) [3](https://sachaartemit-my.sharepoint.com/personal/sacha_artemit_nl/_layouts/15/Doc.aspx?sourcedoc=%7B12CA517D-F3CD-4059-AC04-DC015A1799BB%7D&file=sample_submission.csv&action=default&mobileredirect=true)

**Your task:**  
Build a **reproducible ML pipeline** that outputs lower and upper prediction intervals for each property in the test set and publishes the results as an artifact.

Your analysis must remain:
- Fully **objective**  
- Focused on **data patterns**, **models**, and **pipelines**  
- **Non‑interpretative** 

---

## 📦 Deliverables

### 1. Presentation (PowerPoint or equivalent)

Your slide deck includes:

- Objective findings from your EDA  
- Distribution and trends of sale prices  
- Feature importance and model behaviour  
- Interval modelling comparison (quantile regression, conformal prediction, etc.)  
- Interval quality metrics (coverage, sharpness)  
- Technical explanation (data transformations, pipeline, modelling workflow)

---

## 2. Azure DevOps Repository & Pipelines

Your repository must demonstrate strong engineering practices:

- Git workflow using `dev` → `main`  
- Clean folder structure:

/data
/src
/notebooks
/pipelines
/models
/artifacts
/docs
- Clear commit messages  
- Requirements file (requirements.txt / environment.yml)

### Pipeline Requirements

Your pipeline must perform at least:

- Load and clean datasets  
- Perform feature engineering  
- Train prediction interval model  
- Publish model artifacts  
- Provide reproducible logs and transparency  

You may use:
- Fabric notebooks  
- Azure ML Studio pipelines  
- Python scripts executed via DevOps  

---

## 3. Power BI Dashboard 

### A. House Price Trends  
- Distribution of `sale_price`  
- Variability across cities/submarkets  
- Outlier analysis (based on dataset.csv) 

### B. Property Feature Profiles  
Include attributes such as:  
- `sqft`, `sqft_lot`, `beds`, `baths`, `grade`, `condition`  
- `year_built`, `year_reno`  
- `zoning`, `submarket`, `view_*` indicators

### C. Prediction Intervals Visualisation  
- Comparison of predicted lower vs upper bounds  
- Interval width (sharpness)  
- Coverage rate on validation samples  

### D. Data Sources & Assumptions  
- What columns you used  
- Handling of missing values  
- Modelling assumptions  
- Data limitations  

---

## 🧱 Tasks & Expectations

### 1. Data Extraction

You are responsible for:

- Loading and cleaning **dataset.csv** and **test.csv**  
- Normalising and standardising features  
- Encoding categorical variables (zoning, subdivision, condition, submarket, …)  
- Performing outlier checks  
- Engineering new features, such as:
- Property age  
- Ratios (`land_val` / `imp_val`, `sqft` / `sqft_lot`)  
- Log‑transformed versions of skewed columns  
- Environmental/view indicators (`noise_traffic`, `wfnt`, all `view_*` columns) 

---

### 2. DevOps Engineering

You develop:

- A clean and scalable Git repository  
- CI pipeline (linting, tests, data quality checks)  
- CD pipeline that:
- trains the model  
- produces prediction intervals  
- publishes artifacts  

All pipelines must be **reproducible**, **deterministic**, and **well‑logged**.

---

### 3. ML/AI Component

Your model predicts **prediction intervals** for `sale_price`.

#### Allowed modelling techniques
- Quantile Regression (e.g., XGBoost, LightGBM, CatBoost)  
- Conformal Prediction  
- Bayesian methods  
- Bootstrapped ensembles  

#### Expectations
- Fully reproducible workflow  
- Explainable model design  
- Demonstrate coverage & interval width  

---

### 4. Power BI Dashboard Requirements

#### Recommended Pages
1. Price Distributions  
2. Property & Location Profiles  
3. Prediction Interval Forecasting  
4. Coverage Analysis  
5. Data Sources & Assumptions  

#### Example KPIs
- Pinball Loss  
- Interval Coverage  
- Average interval width  
- MAE on validation  

---

## 🧭 Scope Clarifications

### You MUST:
- Stay objective  
- Only describe data patterns  
- Ensure pipeline reproducibility  

### You MUST NOT:
- Make causal claims  
- Provide market interpretations  
- Give subjective conclusions  

---

## 📘 Recommended Indicators

### Property Attributes
- `sqft`, `sqft_lot`, `beds`, `baths`  
- `grade`, `condition`  
- `imp_val / land_val`  

### Location & Environment Features
- `latitude`, `longitude`  
- `city`, `submarket`  
- Zoning types  
- Scenic/environment indicators (`view_rainier`, `view_sound`, etc.) 

### Derived Features
- Property age  
- Ratio‑based features  
- Log transformations  
- Noise/waterfront/renovation indicators  

---

## 🧠 Suggested Analytical Flow (Non‑Prescriptive)

1. Inspect training data  
2. Clean and structure features  
3. Perform feature engineering  
4. Train prediction interval model  
5. Validate coverage and sharpness  
6. Generate prediction intervals for the test set  
7. Build the Power BI dashboard  
9. Prepare presentation  
10. Finalise DevOps pipeline  

---

## 🗂️ Skill Focus

### Data Engineers  
- Ingestion  
- Pipelines  
- Reproducibility  

### Data Analysts  
- EDA  
- Visualisation  
- Objective interpretation  

### ML Engineers  
- Interval modelling  
- Explainability  
- Model evaluation  

### DevOps Engineers  
- CI/CD  
- Repository structure  
- Automation  

---
