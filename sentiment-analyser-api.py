from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from text_preprocessing import preprocess_data
import pickle

class SentimentAnalyserItem(BaseModel):
    text: str

with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def scoring_endpoint(item: SentimentAnalyserItem):
    # Preprocess the text
    preprocessed_text = preprocess_data(item.text)

    # Transform the preprocessed text using the loaded vectorizer
    sample_vector = tfidf_vectorizer.transform([preprocessed_text])

    probabilities = sentiment_model.predict_proba(sample_vector)[0]
    
    # Calculate percentage probabilities for each class
    positive_percentage = probabilities[1] * 100
    negative_percentage = 100 - positive_percentage
    
    return {"positive_percentage":positive_percentage, "negative_percentage":negative_percentage}
