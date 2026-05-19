
from flask import Flask, render_template, request, jsonify
from preprocessing import ArabicTextPreprocessor
from utils import load_object
import numpy as np

app = Flask(__name__)

preprocessor = ArabicTextPreprocessor()
model = load_object('model.pkl')
tfidf = load_object('tfidf_vectorizer.pkl')
encoder = load_object('label_encoder.pkl')
model_comparison = load_object('model_comparison.pkl')

trained_models = {}
try:
    trained_models['LinearSVC'] = load_object('linearsvc_model.pkl')
    trained_models['LogisticRegression'] = load_object('logisticregression_model.pkl')
except Exception:
    pass

try:
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
except ImportError:
    pass

def predict_era_with_probability(poem_text, use_ensemble=False):
    cleaned = preprocessor.clean_text(poem_text)
    vectorized = tfidf.transform([cleaned])
    
    if use_ensemble:
        predictions = []
        probas = []
        for m in trained_models.values():
            if hasattr(m, 'predict_proba'):
                predictions.append(m.predict(vectorized)[0])
                probas.append(m.predict_proba(vectorized)[0])
        
        if len(probas) > 0:
            avg_proba = np.mean(probas, axis=0)
            prediction_idx = np.argmax(avg_proba)
            predicted_era = encoder.inverse_transform([prediction_idx])[0]
            probabilities = []
            for i, era in enumerate(encoder.classes_):
                probabilities.append({'era': era, 'probability': float(avg_proba[i])})
            probabilities = sorted(probabilities, key=lambda x: x['probability'], reverse=True)
        else:
            prediction = model.predict(vectorized)[0]
            predicted_era = encoder.inverse_transform([prediction])[0]
            probabilities = None
    else:
        prediction = model.predict(vectorized)[0]
        predicted_era = encoder.inverse_transform([prediction])[0]
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(vectorized)[0]
            probabilities = []
            for i, era in enumerate(encoder.classes_):
                probabilities.append({'era': era, 'probability': float(probs[i])})
            probabilities = sorted(probabilities, key=lambda x: x['probability'], reverse=True)
    
    return predicted_era, probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None
    model_comparison_dict = model_comparison.to_dict('records') if hasattr(model_comparison, 'to_dict') else []
    
    if request.method == 'POST':
        poem_text = request.form['poem_text']
        use_ensemble = request.form.get('use_ensemble', 'false') == 'true'
        if poem_text.strip():
            prediction, probabilities = predict_era_with_probability(poem_text, use_ensemble=use_ensemble)
    
    return render_template('index.html', 
                         prediction=prediction, 
                         probabilities=probabilities,
                         model_comparison=model_comparison_dict)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    poem_text = data.get('poem_text', '')
    use_ensemble = data.get('use_ensemble', False)
    
    if not poem_text.strip():
        return jsonify({'error': 'No poem text provided'}), 400
    
    prediction, probabilities = predict_era_with_probability(poem_text, use_ensemble=use_ensemble)
    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)

