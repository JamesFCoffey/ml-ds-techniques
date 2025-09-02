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
│   ├── digit-recognizer.ipynb
│   ├── flower_classification.ipynb
│   ├── forecasting_store_sales.ipynb
│   ├── house_prices.ipynb
│   ├── llm_classification_finetuning.ipynb
│   ├── nlp_disaster_tweets.ipynb
│   ├── spaceship_titanic.ipynb
│   └── titanic.ipynb
├── src/
│   ├── art_creation_gan/                     ← CycleGAN (models, losses, io)
│   ├── contradiction_detection/              ← HF transformers helpers
│   ├── digit_recognizer/                     ← MNIST CNN + tuner
│   ├── flower_classification/                ← EfficientNet utilities
│   ├── forecasting_store_sales/              ← TS features + hybrid predict
│   ├── house_prices/                         ← preprocessing + objectives
│   ├── spaceship_titanic/                    ← features, training, tuning
│   └── titanic/                              ← preprocessing + hyperopt
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

* **notebooks/**: Self-contained Jupyter analyses.
* **src/<project>/**: Project utilities (feature engineering, modeling, tuning)
  imported by notebooks.
* **requirements.txt**: Runtime dependencies.
* **pyproject.toml**: Packaging metadata (editable install via `pip -e .`).
* **LICENSE**: MIT.

---

## 📓 Notebooks & Citations

Each notebook corresponds to a Kaggle competition. Please cite the original
challenge when referencing these solutions.

### House Prices Prediction

* **notebook**: `house_prices.ipynb`
* **competition**: House Prices - Advanced Regression Techniques

  * Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### Titanic

* **notebook**: `titanic.ipynb`
* **competition**: Titanic: Machine Learning from Disaster

  * Kaggle. Titanic: Machine Learning from Disaster.
    [https://kaggle.com/competitions/titanic](https://kaggle.com/competitions/titanic),
    2012. Kaggle.

### Digit Recognizer

* **notebook**: `digit-recognizer.ipynb`
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
   # optional: develop install to import from src/
   pip install -e .
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
