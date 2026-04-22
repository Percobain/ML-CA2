---
title: Student Performance Prediction
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Student Performance Prediction System

A machine learning system that predicts whether a student will pass or fail an exam based on socioeconomic score, daily study hours, daily sleep hours, and class attendance percentage. Built with a Decision Tree classifier, served through a Gradio web interface, and deployed on Hugging Face Spaces.

Live demo: https://huggingface.co/spaces/percobain/Student-Performance-Prediction

## Table of Contents

- [Student Performance Prediction System](#student-performance-prediction-system)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Dataset](#dataset)
  - [Notebook Walkthrough](#notebook-walkthrough)
    - [Step 1: Imports and Setup](#step-1-imports-and-setup)
    - [Step 2: Data Loading and Inspection](#step-2-data-loading-and-inspection)
    - [Step 3: Exploratory Data Analysis](#step-3-exploratory-data-analysis)
    - [Step 4: Feature Engineering and Target Creation](#step-4-feature-engineering-and-target-creation)
    - [Step 5: Train Test Split](#step-5-train-test-split)
    - [Step 6: Baseline Decision Tree](#step-6-baseline-decision-tree)
    - [Step 7: Hyperparameter Tuning](#step-7-hyperparameter-tuning)
    - [Step 8: Evaluation](#step-8-evaluation)
    - [Step 9: Model Export](#step-9-model-export)
    - [Step 10: Prediction Demo](#step-10-prediction-demo)
  - [Why Decision Tree and Not Other Algorithms](#why-decision-tree-and-not-other-algorithms)
  - [Why These Feature Engineering Choices](#why-these-feature-engineering-choices)
  - [Why GridSearchCV and Not Random Search or Bayesian](#why-gridsearchcv-and-not-random-search-or-bayesian)
  - [Why F1 Score as the Optimization Metric](#why-f1-score-as-the-optimization-metric)
  - [Why Joblib and Not Pickle](#why-joblib-and-not-pickle)
  - [Why Gradio and Not Streamlit or Flask](#why-gradio-and-not-streamlit-or-flask)
  - [Final Model Performance](#final-model-performance)
  - [How to Run](#how-to-run)
  - [Team](#team)


## Project Structure

```
mlca/
    data/
        data.csv                              # Raw dataset (1388 records)
    models/
        student_performance_model.joblib      # Trained model artifact
    notebooks/
        student_performance_prediction.ipynb  # Full training pipeline
    app.py                                    # Gradio web application
    requirements.txt                          # Python dependencies
    README.md                                 # This file
```


## Dataset

The dataset contains 1388 student records with 5 columns:

| Column              | Type    | Range        | Description                          |
|---------------------|---------|--------------|--------------------------------------|
| Socioeconomic Score | float64 | 0.101 to 1.0 | Normalized socioeconomic background  |
| Study Hours         | float64 | 0.8 to 10.0  | Daily hours spent studying           |
| Sleep Hours         | float64 | 4.8 to 10.0  | Daily hours of sleep                 |
| Attendance (%)      | float64 | 40 to 100    | Percentage of classes attended        |
| Grades              | float64 | 32 to 91     | Final exam score                     |

Key statistics:
- Mean grade: 40.69, median: 35.0
- No missing values, no duplicate rows
- All columns are numeric, no categorical encoding needed


## Notebook Walkthrough

### Step 1: Imports and Setup

Loads all required libraries: pandas and numpy for data handling, matplotlib and seaborn for visualization, scikit-learn for modeling, and joblib for serialization. Warnings are suppressed to keep notebook output clean during grid search.

**Why these libraries:** scikit-learn is the standard for classical ML in Python. It provides consistent APIs across all algorithms, making it straightforward to swap models, tune hyperparameters, and evaluate results without writing boilerplate code. Seaborn is used on top of matplotlib because it produces cleaner statistical plots with less code (the pairplot and heatmap would require significantly more matplotlib calls to replicate).

### Step 2: Data Loading and Inspection

Reads the CSV file and performs basic sanity checks: shape, column types, missing value counts, and duplicate row detection. This step confirmed:
- 1388 rows, 5 columns
- All float64 types (no type conversion needed)
- Zero missing values (no imputation needed)
- Zero duplicates

**Why this matters:** Skipping data inspection is a common source of silent bugs. A single NaN in the wrong place can cause a model to drop rows during training without warning, leading to a smaller training set and potentially biased results. Checking types upfront also prevents surprises where a numeric column was read as a string due to a stray character.

### Step 3: Exploratory Data Analysis

Three visualizations are generated:

1. **Grade distribution histogram with pass threshold line.** This reveals that the grade distribution is right-skewed, with most students scoring below 50. The red vertical line at 50 makes it visually clear where the pass/fail boundary falls and how imbalanced the classes will be.

2. **Correlation heatmap.** Shows pairwise Pearson correlations between all features and the target. Study Hours and Attendance show the strongest positive correlation with Grades. Sleep Hours shows near-zero correlation, which is expected because both too little and too much sleep can hurt performance (a non-linear relationship that correlation coefficients do not capture).

3. **Pairplot colored by pass/fail.** Scatterplots of every feature pair, colored green for pass and red for fail. This reveals that passing students cluster in the high-study, high-attendance region, confirming that these two features carry the most predictive signal. It also shows that socioeconomic score alone does not cleanly separate the classes, but it does shift the boundary.

**Why EDA before modeling:** Without EDA, you risk building a model on assumptions that do not hold. For example, if Grades had a bimodal distribution, a single threshold of 50 might not be the right way to define pass/fail. The pairplot also confirms that the classes are separable in feature space, which means a classifier has something to learn.

### Step 4: Feature Engineering and Target Creation

**Target variable:** A binary column `Pass` is created where Grades >= 50 maps to 1 (Pass) and everything below maps to 0 (Fail). The resulting class distribution is imbalanced: roughly 80.4% fail, 19.6% pass. This imbalance is important and directly influenced the choice of evaluation metric (F1 over accuracy, discussed later).

**Why 50 as the threshold:** 50 is the standard passing mark in most academic grading systems. Using a data-driven threshold (like the median at 35) would not reflect real-world pass/fail semantics. The goal of this system is to predict a meaningful outcome, not just to split data at an arbitrary point.

**Engineered features:** Four interaction and polynomial features are added:

| Feature                      | Formula                              | Rationale                                                    |
|------------------------------|--------------------------------------|--------------------------------------------------------------|
| Study_x_Attendance           | Study Hours * Attendance (%)         | A student who studies 8h but attends 40% of classes is different from one who studies 8h and attends 90%. The interaction captures this combined effect. |
| Study_x_Socioeconomic        | Study Hours * Socioeconomic Score    | Higher socioeconomic background may amplify the returns on study time (access to better resources, tutoring, quiet study spaces). |
| Attendance_x_Socioeconomic   | Attendance (%) * Socioeconomic Score | Similar reasoning: the benefit of attending class may vary with background. |
| Study_sq                     | Study Hours ^ 2                      | Captures diminishing or accelerating returns on study time. Studying 2h vs 4h is a bigger jump than 8h vs 10h. |

**Why interaction features for a Decision Tree:** Decision trees can only split on one feature at a time. Without interaction features, the tree would need multiple sequential splits to approximate the relationship "high study AND high attendance leads to passing." By providing the product directly, the tree can capture this in a single split, leading to a shallower and more generalizable tree.

**Why not more features:** Adding too many engineered features increases the risk of overfitting, especially with a relatively small dataset. The four features chosen are grounded in domain knowledge about how academic performance works. Features like Sleep_x_Attendance or three-way interactions were not added because there is no clear domain justification for them, and they would increase dimensionality without a clear benefit.

### Step 5: Train Test Split

The data is split 80/20 using stratified sampling:
- Training set: 1110 samples
- Test set: 278 samples
- Train pass rate: 19.6%, Test pass rate: 19.4%

**Why stratified splitting:** With only 19.6% positive cases, a random split could easily produce a test set with a very different pass rate than the training set. Stratification ensures both sets have nearly identical class proportions, making evaluation results more reliable and reproducible.

**Why 80/20 and not 70/30 or 90/10:** With 1388 samples, 80/20 gives 278 test samples, which is enough to get stable evaluation metrics. A 70/30 split would leave only 971 training samples, reducing the model's ability to learn. A 90/10 split would give only 139 test samples, making metrics more volatile (a single misclassification would swing accuracy by nearly 1%).

**Why not cross-validation only:** Cross-validation is used during hyperparameter tuning (Step 7), but a held-out test set is still needed. Cross-validation scores during tuning are optimistic because the tuning process selects the parameters that scored best across folds, which introduces selection bias. The held-out test set provides an unbiased estimate of real-world performance.

### Step 6: Baseline Decision Tree

A default DecisionTreeClassifier (no hyperparameter constraints) is trained to establish a performance floor:
- Accuracy: 97.12%
- Precision: 91.07%
- Recall: 94.44%
- F1 Score: 92.73%
- Tree depth: 10, Leaves: 26

**Why start with a baseline:** Without a baseline, you cannot tell whether hyperparameter tuning actually improved anything. If the tuned model scores 96% accuracy, that sounds good in isolation, but the baseline already achieves 97.12%. The baseline also reveals that this problem is relatively easy for a decision tree, which means aggressive tuning is unlikely to produce dramatic improvements but may improve generalization.

### Step 7: Hyperparameter Tuning

GridSearchCV is run over 5,880 parameter combinations (5-fold stratified CV, totaling 29,400 model fits):

| Parameter         | Values Searched                          |
|-------------------|------------------------------------------|
| max_depth         | 3, 4, 5, 6, 7, 8, 10, 12, 15, None     |
| min_samples_split | 2, 3, 5, 7, 10, 15, 20                  |
| min_samples_leaf  | 1, 2, 3, 5, 7, 10, 15                   |
| criterion         | gini, entropy                            |
| max_features      | sqrt, log2, None                         |
| splitter          | best, random                             |

Best parameters found:
- criterion: gini
- max_depth: 10
- max_features: None (use all features)
- min_samples_leaf: 3
- min_samples_split: 10
- splitter: random

Best CV F1 score: 0.9280

**Why max_depth=10:** The baseline tree also had depth 10, but the tuned tree adds regularization through min_samples_leaf=3 and min_samples_split=10. These constraints prevent the tree from creating tiny leaves that overfit to noise. The depth limit itself is not the primary regularizer here.

**Why splitter=random:** The random splitter does not evaluate every possible threshold for each feature at each node. Instead, it samples random thresholds, which acts as a form of regularization. It introduces controlled randomness that can prevent overfitting on small datasets, similar in spirit to how Random Forests work.

**Why criterion=gini over entropy:** Gini impurity and entropy usually produce very similar trees. Gini is computationally cheaper because it avoids the logarithm calculation. In this case, the grid search confirmed that gini performed as well as entropy, so gini is preferred for efficiency.

### Step 8: Evaluation

The tuned model is evaluated on the held-out test set with multiple metrics:

**Classification Report:**
```
              precision    recall  f1-score   support
        Fail       0.97      0.99      0.98       224
        Pass       0.94      0.89      0.91        54
    accuracy                           0.97       278
```

**Confusion Matrix:** Visualized as a heatmap. Out of 278 test samples, the model misclassified 9: 6 false negatives (predicted fail, actually passed) and 3 false positives (predicted pass, actually failed).

**ROC Curve:** AUC = 0.977, indicating excellent discrimination between classes across all probability thresholds.

**Feature Importance:** The tree's Gini importance scores show which features contribute most to splitting decisions. Study_x_Attendance and Study_x_Socioeconomic rank highest, confirming that the engineered interaction features capture meaningful patterns the raw features alone do not.

**Decision Tree Visualization:** The top 4 levels of the tree are plotted and printed as text rules. The root split is on Study_x_Attendance at 650.06, meaning students with (Study Hours * Attendance) above 650 are classified as passing immediately. This is interpretable: a student studying 8 hours with 82% attendance exceeds this threshold.

**5-Fold Cross-Validation on Training Data:**
- F1: 0.9280 (+/- 0.0898)
- Accuracy: 0.9712 (+/- 0.0359)

The variance across folds is moderate (F1 ranges from 0.854 to 0.989), which is expected with an imbalanced dataset of this size. The minority class (pass) has few samples per fold, so a handful of misclassifications can swing the F1 score significantly.

### Step 9: Model Export

The trained model is saved as a joblib file along with metadata:
- The model object itself
- Feature column names (to ensure correct column ordering at inference time)
- Pass threshold (50)
- Best hyperparameters
- Test set accuracy, F1, and ROC AUC

A verification step immediately reloads the file and confirms the loaded model produces identical predictions on the test set.

**Why save metadata alongside the model:** Saving just the model object is fragile. If the feature columns change order, or if someone forgets which threshold was used, the model will silently produce wrong predictions. Bundling everything into a single artifact makes the model self-documenting.

### Step 10: Prediction Demo

Two sample predictions are run to sanity-check the saved model:
- High-performing student (socioeconomic=0.8, study=8h, sleep=7.5h, attendance=85%) is predicted PASS with 100% confidence.
- Low-performing student (socioeconomic=0.3, study=2h, sleep=6h, attendance=40%) is predicted FAIL with 100% confidence.

These match intuition and confirm the model loaded correctly.


## Why Decision Tree and Not Other Algorithms

The decision to use a Decision Tree was deliberate, not a default. Here is how it compares to the alternatives considered:

**Logistic Regression** assumes a linear relationship between features and the log-odds of passing. The pairplot and correlation analysis show that the boundary between pass and fail is not a simple hyperplane. Study hours below 6 almost always result in failure regardless of other factors, but above 6 the relationship changes character. Logistic regression would need extensive manual polynomial feature engineering to capture this, and even then it would struggle with the sharp thresholds visible in the data.

**Random Forest** is an ensemble of decision trees. It typically outperforms a single tree on complex datasets because it reduces variance through bagging. However, on this dataset (1388 rows, 4 base features), a single well-tuned tree already achieves 96.8% accuracy and 0.977 AUC. A random forest would add computational cost and destroy interpretability without meaningful accuracy gains. The primary value of this system is that a teacher or administrator can look at the tree rules and understand exactly why a student was predicted to fail.

**Support Vector Machine (SVM)** works well for high-dimensional data with clear margins between classes. This dataset is low-dimensional (4 features + 4 engineered), and the decision boundary is axis-aligned (the tree splits on individual features). SVMs are also black-box models, requiring kernel trick visualization to interpret. For a prediction system meant to give actionable feedback ("increase study hours to X"), interpretability is critical.

**Neural Networks** require significantly more data to train without overfitting. With only 1388 samples and 8 features, a neural network would memorize the training set rather than learn generalizable patterns. They also provide no feature importance or decision rules, making it impossible to explain predictions.

**Gradient Boosted Trees (XGBoost, LightGBM)** are the strongest general-purpose classifiers for tabular data. They would likely match or slightly exceed the decision tree's performance. However, they produce ensembles of hundreds of trees, eliminating the interpretability advantage. For this specific use case, where the output needs to explain why a student is at risk and what they can change, a single interpretable tree is more valuable than a marginal accuracy improvement.

**In summary:** The decision tree was chosen because it matches the dataset characteristics (small, tabular, low-dimensional, axis-aligned decision boundaries) and the project requirements (interpretable predictions with actionable explanations). It achieves 96.8% accuracy, which is within 1-2% of what any algorithm would achieve on this data.


## Why These Feature Engineering Choices

**Interaction features (multiplication):** Decision trees split on one feature at a time. To learn "high study AND high attendance leads to passing," the tree needs two sequential splits. By providing the product, the tree can learn this in one split. This produces a shallower tree (fewer nodes, less overfitting) that generalizes better.

**Polynomial feature (Study_sq):** Study hours has the strongest correlation with grades. The squared term lets the tree capture non-linear effects: the difference between 2 and 4 hours of studying is more impactful than the difference between 8 and 10 hours.

**Why not PCA or other transformations:** PCA creates new features that are linear combinations of the originals. These are uninterpretable ("principal component 1" means nothing to a teacher) and decision trees do not benefit from orthogonal features the way linear models do.

**Why not standardization/normalization:** Decision trees are invariant to monotonic feature transformations. Scaling a feature from [0, 1] to [0, 100] does not change where the tree places its splits. StandardScaler is imported in the notebook but intentionally not used because it would add complexity without any benefit.


## Why GridSearchCV and Not Random Search or Bayesian

**GridSearchCV** exhaustively evaluates every combination in the parameter grid. With 5,880 combinations and 5 folds, this means 29,400 model fits. Decision trees train in milliseconds, so the total search completes in under a minute on a modern machine.

**RandomizedSearchCV** samples a fixed number of random combinations from the grid. It is preferred when the search space is large and individual model fits are expensive (e.g., deep learning, large datasets). With only 5,880 combinations and sub-millisecond training times, there is no reason to sample randomly and risk missing the optimal combination.

**Bayesian optimization (Optuna, HyperOpt)** builds a probabilistic model of the parameter space and focuses evaluation on promising regions. This is valuable when the search space has hundreds of thousands of combinations or when each evaluation takes minutes. For 5,880 fast evaluations, the overhead of maintaining a surrogate model is not justified.

**In short:** Grid search is the right tool when the search space is small enough to enumerate exhaustively and each evaluation is cheap. Both conditions hold here.


## Why F1 Score as the Optimization Metric

The dataset is imbalanced: 80.4% fail, 19.6% pass. A model that predicts "fail" for every student would achieve 80.4% accuracy, which sounds reasonable but is completely useless.

**Accuracy** rewards correct predictions equally across classes. With class imbalance, it is dominated by the majority class and does not penalize failure to detect the minority class.

**F1 Score** is the harmonic mean of precision and recall. It requires the model to both find most of the passing students (recall) and be correct when it predicts a pass (precision). Optimizing for F1 forces the model to actually learn the minority class rather than defaulting to the majority prediction.

**Why not AUC:** AUC measures discrimination across all thresholds, but the deployed model uses a fixed threshold (0.5). F1 evaluates performance at the operating threshold, making it more directly relevant to how the model will be used in practice.


## Why Joblib and Not Pickle

Joblib is optimized for serializing objects that contain large numpy arrays. scikit-learn models store their learned parameters as numpy arrays internally. Joblib compresses these arrays more efficiently and deserializes them faster than pickle. It is also the serialization format recommended in scikit-learn's official documentation.

Pickle would work, but joblib produces smaller files and loads faster for this specific type of object.


## Why Gradio and Not Streamlit or Flask

**Gradio** is purpose-built for ML model demos. It provides slider inputs, plot outputs, and example tables with minimal code. The entire app is 242 lines including chart generation logic. Deploying to Hugging Face Spaces requires only pushing the code; Gradio Spaces are a first-class integration on the platform.

**Streamlit** is a general-purpose data app framework. It would require similar amounts of code but does not have native Hugging Face Spaces integration as tight as Gradio. Streamlit also reruns the entire script on every interaction, which would re-load the model and dataset on each slider change.

**Flask** is a web framework, not an ML demo tool. Building the same interface in Flask would require writing HTML templates, JavaScript for sliders, AJAX calls for predictions, and a charting library integration. The amount of code would be 3 to 5 times larger for the same result.


## Final Model Performance

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 96.76% |
| Precision | 94.12% |
| Recall    | 88.89% |
| F1 Score  | 91.43% |
| ROC AUC   | 97.70% |

Best hyperparameters: criterion=gini, max_depth=10, max_features=None, min_samples_leaf=3, min_samples_split=10, splitter=random.


## How to Run

**Install dependencies:**
```
pip install -r requirements.txt
```

**Train the model** (run all cells in the notebook):
```
jupyter notebook notebooks/student_performance_prediction.ipynb
```

**Launch the web app locally:**
```
python app.py
```
Opens at http://localhost:7860

**View the deployed app:**
https://huggingface.co/spaces/percobain/Student-Performance-Prediction


## Team

Shreyans Tatiya - 16010123325

Shreya Menon - 16010123324

Siddhant Raut - 16010123331