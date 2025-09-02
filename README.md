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

Each notebook corresponds to a Kaggle competition (or tutorial). Entries are grouped for quick scanning.

### Tabular ML

#### Titanic – Machine Learning from Disaster

- Notebook: `titanic.ipynb`
- Competition: Titanic – Machine Learning from Disaster
  - Kaggle: https://www.kaggle.com/competitions/titanic
- Summary: Rich tabular feature engineering (titles, surnames, family size,
  cabin/ticket features, imputation, age bucketing, log-fare scaling, rare-cat
  pooling + OHE) with leak-free fit/transform split; compares YDF GBTs and
  XGBoost via Optuna; includes a sklearn wrapper to slot YDF into pipelines.
- Tools & Techniques: pandas/numpy, scikit-learn, Optuna, XGBoost, YDF.

#### House Prices – Advanced Regression Techniques

- Notebook: `house_prices.ipynb`
- Competition: House Prices – Advanced Regression Techniques
  - Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Summary: End-to-end preprocessing with ColumnTransformer, log1p target, and
  model selection across CART, Random Forest, YDF Gradient-Boosted Trees,
  XGBoost, and a compact Keras MLP, tuned with Optuna; returns both sklearn
  arrays and YDF-ready DataFrames for flexible modeling.
- Tools & Techniques: pandas, scikit-learn (ColumnTransformer, OneHotEncoder,
  StandardScaler), Optuna, XGBoost, YDF (TensorFlow Decision Forests), Keras.

#### Spaceship Titanic

- Notebook: `spaceship_titanic.ipynb`
- Competition: Spaceship Titanic
  - Kaggle: https://www.kaggle.com/competitions/spaceship-titanic
- Summary: Custom scikit-learn transformer builds leak-free features (cabin
  parsing, group size and statistics, per-person spend ratios, quantile/domain
  bins, rare-cat collapsing, OHE + scaling); tuned CatBoost/LightGBM/XGBoost
  objectives with Optuna and early stopping; optional soft-vote ensembling.
- Tools & Techniques: scikit-learn Pipelines, CatBoost, LightGBM, XGBoost,
  Optuna, OneHotEncoder, StandardScaler.

### Time Series

#### Store Sales – Time Series Forecasting

- Notebook: `forecasting_store_sales.ipynb`
- Competition: Store Sales – Time Series Forecasting
  - Kaggle: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Summary: Deterministic seasonal/trend regressors (statsmodels DP + Fourier),
  AR lags, weekly oil levels/log-returns with safe forward-fill alignment,
  local/regional holidays, compact on-promotion exposures; baseline Ridge model
  tuned with Optuna plus a residual learner for hybrid predictions.
- Tools & Techniques: pandas/numpy, statsmodels DeterministicProcess, scikit-
  learn (ColumnTransformer, Ridge, StandardScaler), Optuna, time-series feature
  engineering (lags, DOW medians, weekly aggregation).

### NLP

#### NLP with Disaster Tweets

- Notebook: `nlp_disaster_tweets.ipynb`
- Competition: Natural Language Processing with Disaster Tweets
  - Kaggle: https://www.kaggle.com/competitions/nlp-getting-started
- Summary: End-to-end text classification workflow for disaster tweet detection;
  see notebook for modeling choices and experiments.
- Tools & Techniques: Text preprocessing and classification stack (details in
  notebook), optionally Transformers/TF.

#### Contradictory, My Dear Watson (NLI)

- Notebook: `contradiction_detection.ipynb`
- Competition: Contradictory, My Dear Watson
  - Kaggle: https://www.kaggle.com/competitions/contradictory-my-dear-watson
- Summary: Transformer-based sequence classification for multilingual NLI; TF
  model loading with PyTorch-weight fallback and tokenizer utilities for
  fixed-length paired inputs; trains/evaluates with TensorFlow backend.
- Tools & Techniques: Hugging Face Transformers (TFAutoModelForSequenceClassification,
  tokenizers), TensorFlow, fixed-length pair tokenization.

#### LLM Classification Finetuning (Tutorial)

- Notebook: `llm_classification_finetuning.ipynb`
- Competition: N/A — general finetuning workflow notebook
- Summary: End-to-end recipe for finetuning a pretrained LLM encoder on a
  custom text classification dataset: tokenization with fixed sequence length,
  model head initialization, mixed-precision training, evaluation, and export.
- Tools & Techniques: Hugging Face Transformers (AutoTokenizer,
  TFAutoModelForSequenceClassification or AutoModelForSequenceClassification),
  TensorFlow/Keras or PyTorch backend, learning-rate scheduling, early stopping.

### Computer Vision

#### Flower Classification with TPUs

- Notebook: `flower_classification.ipynb`
- Competition: Flower Classification with TPUs
  - Kaggle: https://www.kaggle.com/competitions/flower-classification-with-tpus
- Summary: EfficientNet-B3 classifier built under a distribution strategy,
  with dropout and LR tuned via Keras Tuner; TPU-friendly training utilities.
- Tools & Techniques: TensorFlow/Keras, EfficientNetB3, Keras Tuner, TPUStrategy.

### Generative Modeling

#### I’m Something of a Painter Myself (CycleGAN)

- Notebook: `art_creation_gan.ipynb`
- Competition: I’m Something of a Painter Myself
  - Kaggle: https://www.kaggle.com/competitions/gan-getting-started
- Summary: Lightweight CycleGAN implementation with custom training loop,
  PatchGAN discriminators, instance/group norm blocks, identity + cycle losses;
  TFRecord input pipeline with optional augmentation and TPU detection.
- Tools & Techniques: TensorFlow/Keras, CycleGAN (ResNet/U-Net style gens,
  PatchGAN), custom train_step, TFRecords, TPU/strategy utils.

### Game AI

#### ConnectX

- Notebook: `connect_x.ipynb`
- Competition: ConnectX
  - Kaggle: https://www.kaggle.com/competitions/connectx
- Summary: Notebook explores agent strategies for ConnectX; see notebook for
  details on baseline policies and improvements.
- Tools & Techniques: Kaggle Environments API, heuristic search/game-playing
  utilities (details in notebook).

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
