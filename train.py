

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)
from preprocessing import ArabicTextPreprocessor
from utils import save_object
import warnings
warnings.filterwarnings("ignore")

def main():
    print("=" * 80)
    print("        ARABIC POETRY CLASSIFICATION - IMPROVED PIPELINE")
    print("=" * 80)

    # --------------------------
    # STEP 1: Load and Clean Dataset
    # --------------------------
    print("\n[1/8] Loading dataset...")
    data = pd.read_csv('Arabic_Poetry_Dataset.csv')
    initial_count = len(data)
    print(f"  Initial samples: {initial_count}")

    print("\n[2/8] Cleaning dataset...")
    data = data.dropna(subset=['poem_text', 'poet_era']).reset_index(drop=True)
    data = data.drop_duplicates(subset=['poem_text']).reset_index(drop=True)
    preprocessor = ArabicTextPreprocessor()
    data['cleaned_text'] = data['poem_text'].apply(preprocessor.clean_text)
    data = data[data['cleaned_text'].str.len() > 20].reset_index(drop=True)
    final_count = len(data)
    print(f"  Removed {initial_count - final_count} samples")
    print(f"  Final samples: {final_count}")

    # Encode labels
    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['poet_era'])
    num_labels = len(encoder.classes_)
    print(f"  Number of classes: {num_labels}")

    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['cleaned_text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )
    print(f"  Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    # --------------------------
    # STEP 2: Experiment with Different Feature Setups
    # --------------------------
    print("\n[3/8] Experimenting with feature engineering...")
    
    setups = [
        {
            'name': 'Word TF-IDF (1-2 grams)',
            'tfidf': TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=40000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        },
        {
            'name': 'Char TF-IDF (3-5 grams)',
            'tfidf': TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=40000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        }
    ]

    all_results = []

    for setup in setups:
        print(f"\n  --- Setup: {setup['name']} ---")
        
        tfidf = setup['tfidf']
        X_train = tfidf.fit_transform(X_train_text)
        X_test = tfidf.transform(X_test_text)
        print(f"  Feature matrix shape: {X_train.shape}")

        models = {
            'Naive Bayes (Benchmark)': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, C=1.0),
            'LinearSVC (Primary)': LinearSVC(random_state=42, C=1.0)
        }

        trained_models = {}
        setup_results = []

        for name, model in models.items():
            print(f"\n    Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            setup_results.append({
                'Setup': setup['name'],
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"      Accuracy:  {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")

        all_results.extend(setup_results)

    # --------------------------
    # STEP 3: Select Best Setup and Model
    # --------------------------
    print("\n[4/8] Selecting best setup and model...")
    results_df = pd.DataFrame(all_results).sort_values('Accuracy', ascending=False)
    best_row = results_df.iloc[0]
    best_setup_name = best_row['Setup']
    best_model_name = best_row['Model']
    print(f"\n  Best Setup: {best_setup_name}")
    print(f"  Best Model: {best_model_name}")
    print(f"  Best Accuracy: {best_row['Accuracy']:.4f}")

    # --------------------------
    # STEP 4: Retrain Best Model on Full Training Data
    # --------------------------
    print("\n[5/8] Retraining best model...")
    best_setup = next(s for s in setups if s['name'] == best_setup_name)
    tfidf = best_setup['tfidf']
    X = tfidf.fit_transform(data['cleaned_text'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if best_model_name == 'LinearSVC (Primary)':
        best_model = LinearSVC(random_state=42, C=1.0)
    elif best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, C=1.0)
    else:
        best_model = MultinomialNB()

    best_model.fit(X_train, y_train)

    # Also train LinearSVC and LogisticRegression for ensemble
    print("\n[6/8] Training models for ensemble...")
    linear_svc = LinearSVC(random_state=42, C=1.0)
    logistic_reg = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, C=1.0)
    linear_svc.fit(X_train, y_train)
    logistic_reg.fit(X_train, y_train)

    # --------------------------
    # STEP 5: Evaluate Best Model
    # --------------------------
    print("\n[7/8] Evaluating best model...")
    y_pred_best = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_best)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_best, average='weighted'
    )
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\n  Final Accuracy:  {accuracy:.4f}")
    print(f"  Final Precision: {precision:.4f}")
    print(f"  Final Recall:    {recall:.4f}")
    print(f"  Final F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=encoder.classes_))

    # --------------------------
    # STEP 6: Save Everything
    # --------------------------
    print("\n[8/8] Saving models and objects...")
    save_object(best_model, 'model.pkl')
    save_object(linear_svc, 'linearsvc_model.pkl')
    save_object(logistic_reg, 'logisticregression_model.pkl')
    save_object(tfidf, 'tfidf_vectorizer.pkl')
    save_object(encoder, 'label_encoder.pkl')
    save_object(results_df, 'model_comparison.pkl')
    print("  All files saved successfully!")

    # --------------------------
    # Print Final Results
    # --------------------------
    print("\n" + "=" * 80)
    print("                       FINAL MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"                   BEST MODEL: {best_model_name}")
    print("=" * 80)
    print("\n" + "=" * 80)
    print("                     TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()

