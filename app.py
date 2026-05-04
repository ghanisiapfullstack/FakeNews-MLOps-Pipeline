from flask import Flask, request, render_template, jsonify
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load Pipeline (TF-IDF + Model)
try:
    with open('pipeline_fake_news.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: File pipeline_fake_news.pkl tidak ditemukan!")

# Pastikan stopwords terdownload di server Flask
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # Menghapus teks dalam kurung kotak
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Menghapus tanda baca
    text = re.sub('\w*\d\w*', '', text) # Menghapus angka
    
    # Deteksi Stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

@app.route('/')
def home():
    # Menampilkan UI 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Error Handling: Cek apakah input ada
        input_data = request.json.get('text')
        if not input_data:
            return jsonify({'error': 'Teks tidak boleh kosong!'}), 400
        
        cleaned_data = clean_text(input_data)
        prediction = model.predict([cleaned_data])[0] # Model Predict Data
        
        result = "Real News" if prediction == 1 else "Fake News" # Resultnya
        return jsonify({
            'status': 'success',
            'prediction': result,
            'text_length': len(input_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)