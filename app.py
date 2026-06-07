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
    
    try:
        # Panggil fungsi prediksi dari model.py
        result = prediksi_air(data)
        
        # Tampilkan hasil prediksi di terminal BE
        print("\n" + "="*50)
        print("HASIL PREDIKSI BARU:")
        print(f"Data Input      : {data}")
        print(f"Status Prediksi : {result.get('status_prediksi')}")
        print(f"Confidence Score: {result.get('confidence_score')}%")
        print(f"Validasi SOP    : {result.get('validasi_sop')}")
        print(f"Pelanggaran     : {result.get('pelanggaran')}")
        print(f"Rekomendasi     : {result.get('rekomendasi')}")
        print(f"Warning         : {result.get('warning')}")
        print("="*50 + "\n")
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Jalankan server Flask
    app.run(debug=True, host='0.0.0.0', port=5000)