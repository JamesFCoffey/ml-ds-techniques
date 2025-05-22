# ML & Data Science Techniques Library

A curated collection of Jupyter notebooks and utility scripts demonstrating core
machine learning and data science workflows. These notebooks originated as
solution submissions for various Kaggle Competitions:
[https://www.kaggle.com/competitions](https://www.kaggle.com/competitions).

## 📂 Repository Structure

```
ml-ds-techniques/
├── notebooks/
│   ├── base_data_science_notebook.ipynb      ← reusable template
│   ├── art_creation_gan.ipynb
│   ├── connect_x.ipynb
│   ├── contradiction_detection.ipynb
│   ├── digit_recognizer.ipynb
│   ├── flower_classification.ipynb
│   ├── forecasting_store_sales.ipynb
│   ├── house_prices.ipynb
│   ├── llm_classification_finetuning.ipynb
│   ├── nlp_disaster_tweets.ipynb
│   ├── spaceship_titanic.ipynb
│   └── titanic.ipynb
├── src/
│   ├── art_creation_gan/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── common/
│   │   ├── data_io.py
│   │   ├── feature_tools.py
│   │   └── model_eval.py
│   ├── connect_x/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── contradiction_detection/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── digit_recognizer/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── flower_classification/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── forecasting_store_sales/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── house_prices/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── llm_classification_finetuning/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── nlp_disaster_tweets/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   ├── spaceship_titanic/
│   │   ├── load_data.py
│   │   ├── featurize.py
│   │   └── train.py
│   └── titanic/
│       ├── load_data.py
│       ├── featurize.py
│       └── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

* **notebooks/**: Self-contained Jupyter analyses—from loading data to model
  evaluation.
* **src/common/**: Generic utilities shared across projects (I/O, feature tools,
  evaluation helpers).
* **src/\<competition\_name>/**: Project-specific modules, so each notebook can
  run independently.
* **requirements.txt**: Exact Python package versions.
* **.gitignore**: Excludes environment files, data dumps, and editor artifacts.

---

## 📓 Notebooks & Citations

Each notebook corresponds to a Kaggle competition. Please cite the original
challenge when referencing these solutions.

### House Prices Prediction

* **notebook**: `house_prices_prediction.ipynb`
* **competition**: House Prices - Advanced Regression Techniques

  * Anna Montoya and DataCanary. House Prices - Advanced Regression Techniques.
    [https://kaggle.com/competitions/house-prices-advanced-regression-techniques](https://kaggle.com/competitions/house-prices-advanced-regression-techniques),
    2016. Kaggle.

### Titanic Data Analysis

* **notebook**: `titanic_data_analysis.ipynb`
* **competition**: Titanic: Machine Learning from Disaster

  * Kaggle. Titanic: Machine Learning from Disaster.
    [https://kaggle.com/competitions/titanic](https://kaggle.com/competitions/titanic),
    2012. Kaggle.

### Digit Recognition Business Overview

* **notebook**: `digit_recognition_business_overview.ipynb`
* **competition**: Digit Recognizer

  * Kaggle. Digit Recognizer.
    [https://kaggle.com/competitions/digit-recognizer](https://kaggle.com/competitions/digit-recognizer),
    2013. Kaggle.

---

## 🚀 Getting Started

1. **Clone the repo**

   ```bash
   git clone git@github.com:<your-username>/ml-ds-techniques.git
   cd ml-ds-techniques
   ```

2. **Set up your environment** (via Mamba or venv)

   * **Mamba** (recommended):

     ```bash
     mamba create -n ml-ds-techniques python=3.12 pip
     mamba activate ml-ds-techniques
     ```
   * **venv**:

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   * **If using Mamba, also do**:

     ```bash
     mamba update --all
     ```

4. **Launch Jupyter Lab**

   ```bash
   jupyter lab notebooks/base_data_science_notebook.ipynb
   ```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/xyz`
3. Commit your changes: `git commit -m "Add xyz"`
4. Push & open a PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.