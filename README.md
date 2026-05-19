
# 🏛️ Arabic Poetry Era Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-0.24+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" alt="Status">
</p>

---

## 📜 Project Overview

A university-level Natural Language Processing (NLP) project that classifies Arabic poems into 9 distinct historical eras using both traditional machine learning and modern transformer approaches.

This project emphasizes practical experimentation, honest evaluation, and academic rigor.

### Key Findings
- **Best Model**: Character-level TF-IDF + Logistic Regression (≈53.5% accuracy)
- **Interesting Result**: Traditional ML pipeline outperformed AraBERT on this specific task
- **Focus**: Realistic engineering, no overhyping, no fake metrics

---

## ✨ Features

- **Professional Arabic Text Preprocessing**: Normalization, diacritics removal, and cleaning
- **Multiple Feature Engineering Approaches**: Word-level TF-IDF, character-level TF-IDF
- **Model Comparison**: Naive Bayes, LinearSVC, and Logistic Regression
- **Transformer Experiment**: AraBERT (kept as research comparison)
- **Beautiful Arabic Web Interface**: Heritage-inspired design with elegant typography
- **Thorough Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices
- **Portfolio-Ready**: Clean code, clear documentation, and realistic results

---

## 🛠️ Technologies

### Core Stack
- **Python 3.8+**
- **scikit-learn** - Machine learning library
- **Flask** - Web application framework
- **pandas & numpy** - Data manipulation
- **matplotlib & seaborn** - Visualization (in notebook)

### NLP Pipeline
- **TF-IDF Vectorization** - Both word and character levels
- **Arabic Text Normalization**
  - أ إ آ → ا
  - ى → ي
  - ؤ → و
  - ئ → ي
  - ة → ه
- **Diacritics & Tatweel Removal**
- **Stopword Filtering**

---

## 📁 Project Structure

```
Arabic-Poetry-Classification/
│
├── app.py                          # Flask web application
├── train.py                        # Traditional ML training pipeline
├── preprocessing.py                # Arabic text preprocessing module
├── utils.py                        # Utility functions (save/load models)
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── notebook.ipynb                  # Final, clean Jupyter notebook
│
├── arabert_experiment/             # Research comparison with Transformers
│   └── train_transformer.py        # AraBERT training script
│
├── templates/                      # HTML templates
│   └── index.html                  # Main web interface
│
└── (models are in .gitignore - generated after training)
```

---

## 📊 Dataset

The dataset contains **75,000+ Arabic poems** from 9 historical eras:

| Era | Arabic Name |
|-----|-------------|
| Pre-Islamic | العصر الجاهلي |
| Islamic | العصر الإسلامي |
| Umayyad | العصر الأموي |
| Abbasid | العصر العباسي |
| Andalusian | العصر الأندلسي |
| Ayyubid | العصر الأيوبي |
| Mamluk | العصر المملوكي |
| Ottoman | العصر العثماني |
| Transitional | المخضرمون |

**Data Cleaning Applied**:
- Removed 1381 duplicate poems
- Removed 1012 poems with missing text
- Removed extremely short poems (<20 characters after cleaning)
- Final dataset size: 73,603 poems

---

## 🔬 NLP Experiments & Results

### Feature Engineering

We experimented with two main approaches:

| Approach | Configuration |
|----------|---------------|
| **Word TF-IDF** | ngram_range=(1,2), max_features=40000, sublinear_tf=True |
| **Character TF-IDF** | analyzer='char_wb', ngram_range=(3,5), max_features=40000, sublinear_tf=True |

### Model Comparison

**Final Results (Character TF-IDF)**:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 53.5% | 54.2% | 53.5% | 49.8% |
| LinearSVC | 52.1% | 51.0% | 52.1% | 50.5% |
| Multinomial Naive Bayes | 44.1% | 54.5% | 44.1% | 34.0% |

### Key Insights

#### Why Character TF-IDF Performed Best
- Arabic morphology is rich and complex
- Character-level features capture sub-word patterns and root structures
- More robust to spelling variations and rare words
- Better generalization across different poetic styles

#### Why AraBERT Underperformed
- Smaller effective dataset size (we used a balanced subset for CPU training)
- Classical Arabic poetry has different linguistic patterns than modern text
- Transformers require more data and compute to shine on this task
- Traditional models are more data-efficient for this specific problem

#### Challenges
- Class imbalance in the dataset
- Some eras have overlapping poetic styles
- Limited labeled historical data
- Subjectivity in era labeling for transitional poets

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
cd "Arabic Poetry Classification"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add the Dataset
The dataset file (Arabic_Poetry_Dataset.csv) is not included in this repository due to size.
Place the dataset file in the root directory of the project.

### Step 4: Train the Models
```bash
python train.py
```

This will:
- Load and clean the dataset
- Preprocess Arabic text
- Experiment with different feature setups
- Train and compare all models
- Save the best model and preprocessing objects

### Step 5: Run the Web Application
```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

---

## 📦 Models & Artifacts

- **Model files (.pkl)**: Generated by train.py, stored locally, ignored by git
- **Dataset (Arabic_Poetry_Dataset.csv)**: Place manually
- **Transformer checkpoints (arabert_model/)**: Training artifacts, ignored by git

To regenerate the model files, simply run `python train.py`!

---

## 🌟 Web Interface

The application features a beautiful, heritage-inspired design:
- Elegant Arabic typography using Amiri and Cairo fonts
- Rich, cultural color palette with gold accents
- Fully RTL (right-to-left) layout
- Responsive design for all devices
- Smooth animations and transitions
- Clear prediction display with probability bars

---

## 🔮 Future Improvements

This is a strong portfolio project, but there's always room for growth:

### Data & Preprocessing
- **Better Dataset Balancing**: Oversample minority classes or use class weights
- **More Historical Data**: Expand the dataset with more classical poetry sources
- **Improved Labeling**: Curate a higher-quality, expert-labeled dataset
- **Advanced Preprocessing**: Experiment with Arabic-specific tokenizers and morphological analyzers

### Modeling
- **Hybrid Models**: Combine character TF-IDF with transformer embeddings
- **Ensemble Methods**: Stack multiple models for better performance
- **Hyperparameter Tuning**: Use Optuna or GridSearchCV for systematic optimization
- **Advanced Arabic Embeddings**: Experiment with AraVec or other pre-trained Arabic embeddings

### Infrastructure
- **GPU Training**: Scale up transformer experiments with GPU acceleration
- **Model Deployment**: Deploy to a cloud platform like Heroku or AWS
- **API Documentation**: Add proper API docs with Swagger/OpenAPI
- **Testing**: Add unit tests and integration tests

---

## 📝 License

This project is for educational purposes and is part of a professional AI/ML portfolio.

---

## 🙏 Acknowledgments

- All poets and scholars who preserved this rich literary heritage
- The open-source NLP community
- Classical Arabic literature enthusiasts

---

<p align="center">
  <b>✨ Built with passion for Arabic poetry and practical AI ✨</b>
</p>

