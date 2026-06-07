from flask import Flask, request, jsonify
from model import prediksi_air   # pastikan file model.py di direktori yang sama

app = Flask(__name__)

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
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Jalankan server Flask
    app.run(debug=True, host='0.0.0.0', port=5000)