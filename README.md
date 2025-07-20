 Voice Review Sentiment & Fake Review Detector
AI-powered web app to detect fake or suspicious product reviews from both text and voice.
Live Demo: fake-review-detector-b6fhdtpv33jakrjmhvjqlh.streamlit.app

 Overview
This open-source app leverages both rule-based logic and machine learning to analyze and flag potentially fake reviews.
It’s designed to help e-commerce platforms, sellers, and buyers build trust by identifying suspicious feedback—whether written or spoken.

 Key Features
 Single Review Analysis:
Enter a review as text or upload/record an audio review for instant analysis.

Voice Review Support:
Transcribes audio reviews using Whisper AI and analyzes them like text.

 Bulk Review Upload:
Upload a CSV file of reviews for batch analysis and download results.

 ML-Based Prediction:
Uses a trained Logistic Regression model to predict review authenticity.

 Explainable Results:
See a genuineness score, rule violation breakdown, and pie chart visualization for every review.

Downloadable Reports:
Export analyzed results as a CSV file.

 Live Demo
[Click here to try the app on Streamlit Cloud](https://fake-review-detector-b6fhdtpv33jakrjmhvjqlh.streamlit.app/)


 Tech Stack
Component	 ->Purpose
Python ->	Core programming language
Streamlit	->Web app framework
OpenAI Whisper	->Speech-to-text (audio reviews)
TextBlob->	Sentiment analysis
Scikit-learn->	Machine learning (LogReg)
Pandas	->Data handling
Matplotlib	->Visualization (pie charts)
Deep Translator->	Multilingual support
streamlit-mic-recorder->	Microphone input
 How to Use
Single Review:
Enter text or upload/record audio, set a star rating, and click "Analyze" for instant feedback.

Bulk Review:
Upload a CSV file with a review column to analyze multiple reviews at once.

ML Prediction:
Enter a review in the ML tab to get a machine learning-based authenticity prediction.
 Installation
Clone the repository:
git clone https://github.com/Nitin369-star/fake-review-detector.git
cd fake-review-detector

Install Python dependencies:
pip install -r requirements.txt

Install system dependencies (FFmpeg required for audio):

Linux: sudo apt install ffmpeg
Windows: Download FFmpeg and add to PATH

Run the app:
streamlit run app.py
 Screenshots
![image](https://github.com/user-attachments/assets/882b917e-71d1-40f3-b533-9690fb801295)
![image](https://github.com/user-attachments/assets/f2caee4c-4be1-4764-a0a7-2882d4a805e0)



 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

 License
MIT

Made  by Nitin Negi
DTU, 2nd Semester | AI & Machine Learning Enthusiast



