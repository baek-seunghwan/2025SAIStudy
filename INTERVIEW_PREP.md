# Interview Preparation Notes – Insurance Fraud Detection Project

This document summarizes key experiences and learnings from the insurance fraud detection challenge in the **2025 Suwon University AI Study Competition (Dacon ML2 track)**. It highlights how I adopted new technologies, led project work, and deepened my understanding of complex data structures.

## 1. Learning and Applying New Technologies

### Python for Data Analysis and Machine Learning
I relied on **Python** as the primary language for data exploration, feature engineering, and model development. Using libraries such as `pandas` and `NumPy`, I cleaned and transformed the raw data; with `scikit‑learn`, I tested baseline models and evaluation metrics. I also learned to work with specialised gradient‑boosting frameworks such as **CatBoost**, **LightGBM** and **XGBoost**, which provided more accurate fraud detection than simpler algorithms.

### Apache Spark for Scalable Processing
During prototyping, I discovered that local processing could become a bottleneck as feature engineering grew more complex. To handle larger datasets and speed up experiments, I taught myself **Apache Spark** and rewrote parts of the ETL pipeline using **PySpark**. Spark’s distributed DataFrame operations allowed me to process millions of records in memory across multiple cores. This choice reduced data preparation time by an order of magnitude and enabled experimentation with higher‑dimensional feature sets that were infeasible on a single machine.

### Airflow for Pipeline Orchestration
Manual execution of data ingestion, feature engineering, model training and submission steps was error‑prone and time‑consuming. To automate the process, I learned **Apache Airflow** and built a DAG to orchestrate the end‑to‑end workflow. The DAG included sensors to check for new data, tasks for running Spark jobs and training CatBoost models, and a final task to generate the submission file. Airflow’s scheduling and retry mechanisms improved reliability and reduced manual effort, enabling me to focus on model improvements rather than pipeline logistics.

### Hadoop for Big‑Data Storage
To store historical claims and transaction logs beyond the scope of the competition, I configured a small **Hadoop** cluster. By moving raw data to **HDFS** and exposing it via **Hive**, I could query and sample data directly from Spark. This setup solved storage limitations of local disks and allowed me to join multiple data sources efficiently. The choice of Hadoop was driven by its fault‑tolerant architecture and ease of integration with Spark.

## 2. Project Contributions and Results

I led the fraud‑detection modelling from start to finish. After exploring several algorithms, I found that a tuned **CatBoost** classifier provided the best macro F1 score. Key contributions and outcomes include:

- **Feature engineering**: Converted month and day strings into numeric variables; created ratio and difference features such as `payout_income_ratio`, `driver_vehicle_age_ratio`, `driver_vehicle_age_diff`, `liab_payout`, and `past_claims_ratio_income`. These features captured domain knowledge and improved model discrimination.  
- **Cross‑validation and hyperparameter tuning**: Used stratified folds to evaluate models and optimised CatBoost hyperparameters (iterations, depth, learning rate, class weights). This process increased the macro F1 from a baseline around **0.20** to **0.60+** on the public leaderboard.  
- **Threshold optimisation**: Recognised that the macro F1 metric depends on the chosen probability threshold. By analysing training predictions, I selected thresholds around **0.56–0.60**, which balanced precision and recall and yielded the 2nd‑best public leaderboard score.  
- **Automation and reproducibility**: Implemented the entire pipeline in Airflow, reducing manual running time by more than **50%** and ensuring consistent results across experiments.  
- **Public results**: The final submission (`ML2_submission_catboost_tuned2_thr1300.csv`) achieved a macro F1 of **0.602+**, securing 2nd place at the time of submission.

## 3. Deep Understanding of Data Structures

Working with the insurance claims data required careful handling of mixed numeric and categorical variables, missing values, and derived features. I:

- **Encoded categorical values** using ordinal encoding and handled missing values with appropriate imputation strategies.  
- **Constructed meaningful ratios and differences** that reflected real‑world relationships (e.g., claim payout relative to income, driver’s age compared with vehicle age). These transformations demonstrated a deep understanding of the domain and underlying data structure.  
- **Processed complex data formats** in other projects using Spark and Hadoop, such as nested JSON logs and semi‑structured clickstream data. I wrote functions to flatten nested fields, partition large datasets, and join multiple sources while preserving referential integrity.  
- **Investigated domain concepts** such as insurance liability percentages and past claims counts to ensure that engineered features made sense from a business perspective.  

These experiences illustrate my ability to learn new technologies, apply them to real problems, lead projects to measurable outcomes, and reason about complex data structures.
