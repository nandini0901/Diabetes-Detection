import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('models/diabetes_model.keras')

# Load original data to fit the scaler
df = pd.read_csv('data/diabetes.csv', header=None)
df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
              'Insulin','BMI','DiabetesPedigree','Age','Outcome']

X = df.drop('Outcome', axis=1).values
scaler = StandardScaler()
scaler.fit(X)

# Example patient data - you can change these values!
patient = np.array([[2, 120, 70, 25, 100, 28.5, 0.350, 30]])

patient_scaled = scaler.transform(patient)
prediction = model.predict(patient_scaled)
probability = prediction[0][0]

print(f"\nDiabetes Probability: {probability*100:.2f}%")
if probability > 0.5:
    print("Result: ⚠️  HIGH RISK - Likely Diabetic")
else:
    print("Result: ✅  LOW RISK - Likely Not Diabetic")
