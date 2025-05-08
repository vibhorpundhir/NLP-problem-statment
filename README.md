https://colab.research.google.com/drive/1NbftaPG8WJUDn-POJuyr5-bf28wfrf9L?usp=sharing
VIBHOR PUNDHIR (RA2211027010031)
CSE BIG DATA 

Real-Time QoS Optimization for Live Game Streaming
📌 Problem Statement
As live video game streaming platforms like Twitch and YouTube Gaming grow rapidly, they face challenges in delivering consistent Quality of Service (QoS) due to unpredictable network conditions, fluctuating stream quality, and varying viewer engagement. Traditional QoS techniques are reactive and do not adapt well to real-time user experience and sentiment.

✅ Solution
This project presents a data-driven, real-time QoS optimization system that integrates:
Big Data Analytics: To process and simulate streaming-related metrics like buffering time, watch duration, and video quality.
BERT-based Sentiment Analysis: To understand user engagement and feedback from live chat messages.
Deep Neural Network (DNN): To predict viewer satisfaction based on QoS indicators and sentiment.
Dynamic QoS Decision Logic: To take real-time actions such as switching bitrate or maintaining quality based on satisfaction probability.

🧠 Features
Cleans and processes streaming data
Analyzes chat sentiment using HuggingFace's DistilBERT
Trains a DNN to classify viewer satisfaction
Uses satisfaction probability to simulate real-time QoS decisions
Calculates precision, recall, F1-score, and accuracy for model evaluation
Generates tabular results for analysis

## 🧰 Technologies Used

- Python (3.8+)
- Google Colab / Jupyter Notebook
- PyTorch
- HuggingFace Transformers (BERT)
- Scikit-learn
- Pandas / NumPy / Matplotlib


📊 Sample Outputs
Buffering Time	Video Quality	Watch Duration	Sentiment	Predicted Satisfaction	Action
5.3 sec	480p	1200 sec	POSITIVE	0.84	Maintain stream quality
9.1 sec	360p	300 sec	NEGATIVE	0.32	Switch to lower bitrate


