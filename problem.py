# ===============================
# 1. Install Required Libraries
# ===============================
!pip install transformers --quiet

# ===============================
# 2. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline
import matplotlib.pyplot as plt

# ===============================
# 3. Generate Synthetic Dataset
# ===============================
np.random.seed(42)
num_samples = 1000

df = pd.DataFrame({
    "buffering_time": np.random.uniform(0, 10, size=num_samples),
    "video_quality": np.random.choice(["240p", "360p", "480p", "720p", "1080p"], size=num_samples),
    "watch_duration": np.random.uniform(60, 7200, size=num_samples),
    "chat_message": np.random.choice(
        ['Good stream', 'Laggy video', 'Great quality', 'Buffering issues', 'Nice', 'So boring', 'Excellent', 'Too much lag'],
        size=num_samples
    ),
    "satisfaction_score": np.random.choice([0, 1], size=num_samples)
})

# ===============================
# 4. Clean Text
# ===============================
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned_message'] = df['chat_message'].apply(clean_text)

# ===============================
# 5. Sentiment Analysis using BERT
# ===============================
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentiments = sentiment_pipeline(df['cleaned_message'].tolist(), batch_size=32)
df['sentiment_label'] = [x['label'] for x in sentiments]
df['sentiment_score'] = [x['score'] for x in sentiments]

# ===============================
# 6. Video Quality Mapping
# ===============================
quality_mapping = {"240p": 1, "360p": 2, "480p": 3, "720p": 4, "1080p": 5}
df["video_quality_num"] = df["video_quality"].map(quality_mapping)

# ===============================
# 7. Prepare Features & Normalize
# ===============================
features = df[["buffering_time", "video_quality_num", "watch_duration", "sentiment_score"]]
labels = df["satisfaction_score"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# ===============================
# 8. Define and Train DNN Model
# ===============================
class SatisfactionDNN(nn.Module):
    def __init__(self, input_dim):
        super(SatisfactionDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SatisfactionDNN(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

loss_history = []
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ===============================
# 9. Evaluation and Accuracy
# ===============================
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()

# Accuracy
accuracy = accuracy_score(y_test_tensor, predicted_classes)
print(f"\n✅ Test Accuracy: {accuracy:.2f}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test_tensor, predicted_classes))

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test_tensor, predicted_classes))

# ===============================
# 10. Real-Time QoS Decision Example
# ===============================
def simulate_qos_decision(viewer_row):
    input_vector = np.array([[viewer_row["buffering_time"],
                              quality_mapping[viewer_row["video_quality"]],
                              viewer_row["watch_duration"],
                              viewer_row["sentiment_score"]]])
    input_scaled = scaler.transform(input_vector)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        satisfaction_prob = model(input_tensor).item()

    if satisfaction_prob < 0.5:
        action = "⚠️ Switch to lower bitrate to reduce buffering"
    else:
        action = "✅ Maintain current stream quality"

    return satisfaction_prob, action

sample_viewer = df.iloc[10]
satisfaction_prob, action = simulate_qos_decision(sample_viewer)
print(f"\n🧪 Sample Viewer Satisfaction: {satisfaction_prob:.2f}, Action: {action}")

# ===============================
# 11. Tabular Analysis Output
# ===============================
# Summary Stats
print("\n📊 Summary Statistics:")
display(df.describe())

# Sentiment Distribution
print("\n📊 Sentiment Label Distribution:")
display(df['sentiment_label'].value_counts())

# Average Satisfaction per Quality
print("\n📊 Avg Satisfaction by Video Quality:")
display(df.groupby('video_quality')['satisfaction_score'].mean())

# Add predictions to dataframe
df['predicted_satisfaction'] = (model(torch.tensor(features_scaled, dtype=torch.float32)) > 0.5).float()

# Sample Prediction Table
print("\n📊 Sample Predictions Table:")
display(df[['buffering_time', 'video_quality', 'watch_duration', 'sentiment_score',
            'satisfaction_score', 'predicted_satisfaction']].head(10))
