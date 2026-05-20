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
        input_data = request.json.get('text')
        if not input_data:
            return jsonify({'error': 'Teks tidak boleh kosong!'}), 400
        
        cleaned_data = clean_text(input_data)
        prediction = model.predict([cleaned_data])[0] 
        
        # Ambil probabilitas persentase
        probabilities = model.predict_proba([cleaned_data])[0]
        confidence_score = max(probabilities) * 100 
        
        # Tetapkan hasil asli tanpa persen dulu agar JS lama tidak rusak
        result = "Real News" if prediction == 1 else "Fake News"
        
        return jsonify({
            'status': 'success',
            'prediction': result,          # Tetap "Real News" atau "Fake News" (JS kamu aman)
            'confidence': f"{confidence_score:.2f}%", # Variabel baru untuk persentase
            'text_length': len(input_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500