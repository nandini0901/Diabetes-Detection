# 🩺 Diabetes Detection using Neural Network

A machine learning project that predicts whether a patient has diabetes or not using a Neural Network built with TensorFlow/Keras.

## 📊 Dataset
- **Source:** Pima Indians Diabetes Dataset
- **Total Records:** 768 patients
- **Features:** 8 medical input variables
- **Target:** Diabetic (1) or Not Diabetic (0)

## 🧠 Features Used
| Feature | Description |
|--------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigree | Diabetes pedigree function |
| Age | Age in years |

## 🏗️ Project Structure
```
Diabetes-Detection/
└── diabetes_detection/
    ├── data/
    │   └── diabetes.csv        # Dataset
    ├── models/
    │   └── diabetes_model.keras # Saved model
    ├── main.py                  # Training script
    ├── predict.py               # Prediction script
    └── requirements.txt         # Dependencies
```

## 🔧 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/nandini0901/Diabetes-Detection.git
cd Diabetes-Detection/diabetes_detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python main.py
```

### 4. Make predictions
```bash
python predict.py
```

## 📈 Model Performance
- **Algorithm:** Neural Network (TensorFlow/Keras)
- **Test Accuracy:** 77.92%
- **Precision (Non-Diabetic):** 81%
- **Precision (Diabetic):** 71%

## 🏛️ Model Architecture
```
Input Layer  → 8 features
Hidden Layer 1 → 16 neurons (ReLU) + Dropout(0.2)
Hidden Layer 2 → 8 neurons (ReLU) + Dropout(0.2)
Output Layer → 1 neuron (Sigmoid)
```

## 🛠️ Tech Stack
- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

## 👩‍💻 Author
**Nandini Agrawal**
