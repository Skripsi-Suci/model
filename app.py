import os
from flask import Flask, request, jsonify
from model import prediksi_air   # pastikan file model.py di direktori yang sama

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data JSON dari request
    data = request.get_json()

    # Validasi parameter wajib
    required = ['pH', 'SC', 'Nitrite', 'Fe', 'Sulfate', 'Turbidity']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing parameters. Required: ' + ', '.join(required)}), 400

    custom_limits = data.get('custom_limits', None)

    # Siapkan data parameter saja (hanya 6 parameter numerik, tanpa custom_limits)
    param_keys = ['pH', 'SC', 'Nitrite', 'Fe', 'Sulfate', 'Turbidity']
    param_data = {k: data[k] for k in param_keys if k in data}

    # Konversi nilai custom_limits dari list ke tuple jika ada
    if custom_limits is not None:
        custom_limits = {k: tuple(v) for k, v in custom_limits.items()}

    try:
        # Panggil fungsi prediksi dari model.py dengan custom_limits
        result = prediksi_air(param_data, custom_limits)

        # Buat response JSON terlebih dahulu sebelum logging ke terminal
        response = jsonify(result)

        # Logging ke terminal (aman untuk Windows — encode karakter non-ASCII)
        def safe(text):
            return str(text).encode('ascii', errors='replace').decode('ascii')

        print("\n" + "="*50)
        print("HASIL PREDIKSI BARU:")
        print("Data Input      : " + safe(param_data))
        print("Status Prediksi : " + safe(result.get('status_prediksi')))
        print("Confidence Score: " + safe(result.get('confidence_score')) + "%")
        print("Validasi SOP    : " + safe(result.get('validasi_sop')))
        print("Pelanggaran     : " + safe(result.get('pelanggaran')))
        print("Rekomendasi     : " + safe(result.get('rekomendasi')))
        print("Warning         : " + safe(result.get('warning')))
        print("="*50 + "\n")

        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# State global untuk pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset_coolingwater.csv")

pipeline_state = {
    "df": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "model": None
}

# CORS headers support
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

def ensure_dataset_loaded():
    if pipeline_state["df"] is None:
        import pandas as pd
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        pipeline_state["df"] = df

def ensure_dataset_preprocessed():
    ensure_dataset_loaded()
    df = pipeline_state["df"]
    if df.duplicated().sum() > 0:
        pipeline_state["df"] = df.drop_duplicates()

def ensure_data_split():
    ensure_dataset_preprocessed()
    if pipeline_state["X_train"] is None:
        from sklearn.model_selection import train_test_split
        df = pipeline_state["df"]
        X = df.drop(["Target", "Target_encoded"], axis=1)
        y = df["Target_encoded"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        pipeline_state["X_train"] = X_train
        pipeline_state["X_test"] = X_test
        pipeline_state["y_train"] = y_train
        pipeline_state["y_test"] = y_test

def ensure_model_trained():
    ensure_data_split()
    if pipeline_state["model"] is None:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf.fit(pipeline_state["X_train"], pipeline_state["y_train"])
        pipeline_state["model"] = rf

@app.route('/api/pipeline/load', methods=['GET', 'POST'])
def pipeline_load():
    try:
        import pandas as pd
        if not os.path.exists(DATASET_PATH):
            return jsonify({"status": "error", "message": f"File dataset tidak ditemukan di: {DATASET_PATH}"}), 404
        df = pd.read_csv(DATASET_PATH)
        pipeline_state["df"] = df
        
        # Reset state lain
        pipeline_state["X_train"] = None
        pipeline_state["X_test"] = None
        pipeline_state["y_train"] = None
        pipeline_state["y_test"] = None
        pipeline_state["model"] = None
        
        class_counts = df["Target"].value_counts().to_dict()
        class_counts = {k: int(v) for k, v in class_counts.items()}
        missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
        
        return jsonify({
            "status": "success",
            "message": "Dataset loaded successfully.",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "class_counts": class_counts,
            "missing_values": missing_values
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/preprocess', methods=['POST'])
def pipeline_preprocess():
    try:
        ensure_dataset_loaded()
        df = pipeline_state["df"]
        
        rows_before = len(df)
        duplicate_count = int(df.duplicated().sum())
        
        df_clean = df.drop_duplicates()
        pipeline_state["df"] = df_clean
        rows_after = len(df_clean)
        
        missing_values = {k: int(v) for k, v in df_clean.isnull().sum().to_dict().items()}
        
        return jsonify({
            "status": "success",
            "message": "Data preprocessed successfully (duplicates dropped).",
            "duplicate_count": duplicate_count,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "missing_values": missing_values
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/split', methods=['POST'])
def pipeline_split():
    try:
        data = request.get_json() or {}
        test_size = float(data.get('test_size', 0.3))
        random_state = int(data.get('random_state', 42))
        
        ensure_dataset_preprocessed()
        df = pipeline_state["df"]
        
        X = df.drop(["Target", "Target_encoded"], axis=1)
        y = df["Target_encoded"]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        pipeline_state["X_train"] = X_train
        pipeline_state["X_test"] = X_test
        pipeline_state["y_train"] = y_train
        pipeline_state["y_test"] = y_test
        
        return jsonify({
            "status": "success",
            "message": "Data split successfully.",
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
            "features": list(X.columns)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/cv', methods=['POST'])
def pipeline_cv():
    try:
        data = request.get_json() or {}
        n_splits = int(data.get('n_splits', 5))
        random_state = int(data.get('random_state', 42))
        
        ensure_data_split()
        X_train = pipeline_state["X_train"]
        y_train = pipeline_state["y_train"]
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='f1')
        
        return jsonify({
            "status": "success",
            "fold_f1_scores": [float(score) for score in cv_scores],
            "mean_f1_score": float(cv_scores.mean())
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/train', methods=['POST'])
def pipeline_train():
    try:
        data = request.get_json() or {}
        random_state = int(data.get('random_state', 42))
        
        ensure_data_split()
        X_train = pipeline_state["X_train"]
        y_train = pipeline_state["y_train"]
        
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        rf.fit(X_train, y_train)
        
        pipeline_state["model"] = rf
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully."
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/evaluate', methods=['GET', 'POST'])
def pipeline_evaluate():
    try:
        ensure_model_trained()
        model = pipeline_state["model"]
        X_test = pipeline_state["X_test"]
        y_test = pipeline_state["y_test"]
        
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = model.predict(X_test)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        formatted_report = {}
        for key, metrics in report_dict.items():
            if isinstance(metrics, dict):
                formatted_report[key] = {k: float(v) if k != 'support' else int(v) for k, v in metrics.items()}
            else:
                formatted_report[key] = float(metrics)
                
        return jsonify({
            "status": "success",
            "accuracy": accuracy,
            "classification_report": formatted_report
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/confusion-matrix', methods=['GET', 'POST'])
def pipeline_confusion_matrix():
    try:
        ensure_model_trained()
        model = pipeline_state["model"]
        X_test = pipeline_state["X_test"]
        y_test = pipeline_state["y_test"]
        
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        
        return jsonify({
            "status": "success",
            "confusion_matrix": [[int(cell) for cell in row] for row in cm],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "classes": ["Tidak Layak", "Layak"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/feature-importance', methods=['GET', 'POST'])
def pipeline_feature_importance():
    try:
        ensure_model_trained()
        model = pipeline_state["model"]
        X_train = pipeline_state["X_train"]
        
        importances = model.feature_importances_
        feature_importance_dict = {
            feature: float(imp) for feature, imp in zip(X_train.columns, importances)
        }
        
        sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return jsonify({
            "status": "success",
            "feature_importance": sorted_importance
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pipeline/save', methods=['POST'])
def pipeline_save():
    try:
        ensure_model_trained()
        model = pipeline_state["model"]
        
        data = request.get_json() or {}
        filename = data.get('filename', 'Randomforest_baru.pkl')
        filename = os.path.basename(filename)
        
        import joblib
        save_path = os.path.join(BASE_DIR, filename)
        joblib.dump(model, save_path)
        
        return jsonify({
            "status": "success",
            "message": f"Model saved successfully as {filename}.",
            "filename": filename
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # CATATAN: Blok ini HANYA untuk pengembangan lokal (development server).
    # Di production (Hugging Face) aplikasi dijalankan oleh Gunicorn,
    # yang langsung memakai objek `app` di atas (lihat Dockerfile: "app:app").
    # Port mengikuti environment var PORT; default 7860 sesuai Hugging Face Spaces.
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)