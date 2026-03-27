import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load Data
df = pd.read_csv('data/diabetes.csv', header=None)
df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
              'Insulin','BMI','DiabetesPedigree','Age','Outcome']

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nOutcome distribution:\n", df['Outcome'].value_counts())

# 2. Prepare Data
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                    validation_split=0.2, verbose=1)

# 5. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
model.save('models/diabetes_model.keras')
print("Model saved!")