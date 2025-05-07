import joblib
from flask import Flask, render_template, request

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('fake_news_model.pkl')  # Path to your trained model
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Path to your TF-IDF vectorizer

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form input
        title = request.form['title']
        text = request.form['text']
        
        # Combine the title and text for prediction
        combined_text = title + " " + text
        
        # Transform the input text using the loaded TF-IDF vectorizer
        text_tfidf = tfidf.transform([combined_text])
        
        # Predict with the loaded model
        prediction = model.predict(text_tfidf)
        
        # Return the result (Fake or Real)
        result = 'Real' if prediction == 'Real' else 'Fake'
        
        return render_template('index.html', prediction_text=f"The article is: {result}")

if __name__ == "__main__":
    app.run(debug=True)
