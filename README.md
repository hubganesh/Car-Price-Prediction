# Car Price Prediction - Machine Learning Project

## 🚗 Overview

A comprehensive machine learning project for predicting used car prices in India using Random Forest Regression. This project demonstrates advanced ML techniques including feature engineering, model training, and web deployment with Streamlit.

## ✨ Features

- **High Accuracy**: R² score > 0.94 with confidence intervals
- **Advanced Features**: Brand, Seats, Car Age, KM/Year as key predictors
- **Interactive Web App**: Streamlit interface for real-time predictions
- **Model Confidence**: Displays prediction intervals and confidence metrics
- **Professional UI**: Clean, user-friendly interface

## 🛠️ Tech Stack

- **Machine Learning**: scikit-learn, Random Forest Regressor
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Web Framework**: Streamlit
- **Deployment**: Ready for Streamlit Cloud/GitHub Pages

## 📊 Dataset

- Source: YBI Foundation Car Selling Price Dataset
- Features: Brand, Year, KM Driven, Fuel Type, Transmission, Owner Type, Seats, Engine, Max Power, Mileage, Torque, Seller Type
- Engineered Features: Car Age, KM per Year

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
streamlit run app.py
```

## 📈 Model Performance

- **R² Score**: 0.94+
- **MAE**: < ₹100,000
- **RMSE**: < ₹200,000
- **Key Features**: Brand (highest impact), Seats, Car Age, KM/Year

## 🎯 Usage

1. Select car brand from dropdown
2. Choose seating capacity
3. Enter manufacturing year and kilometers driven
4. Select fuel type, transmission, and owner type
5. Click "Predict Price" to get estimate with confidence interval

## 🤖 Model Details

### Training Process

1. Data preprocessing and cleaning
2. Feature engineering (Car Age, KM per Year)
3. Categorical encoding (One-hot encoding)
4. Random Forest training (300 estimators, max depth 20)
5. Model evaluation and validation

### Key Features

- **Brand**: Most important feature (3-4x price difference)
- **Seats**: 7/8-seaters have ₹200K-400K premium
- **Car Age**: Each year reduces price by ~10%
- **KM per Year**: Higher usage decreases value

## 📁 Project Structure

```
├── app.py                    # Streamlit web application
├── car price predict.ipynb   # Jupyter notebook for training
├── car_price_model.pkl       # Trained Random Forest model
├── model_columns.pkl         # Feature column names
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore rules
```

## 🔧 Development

### Training the Model

Run the Jupyter notebook `car price predict.ipynb` to:
- Load and preprocess data
- Train the Random Forest model
- Evaluate performance
- Save model and column names

### Web Application

The `app.py` file contains the Streamlit web interface with:
- User input forms
- Data preprocessing pipeline
- Model prediction with confidence
- Results display

## 📊 Results

The model achieves high accuracy with the following metrics:
- R² Score: 0.94+
- Mean Absolute Error: ~₹80,000
- Root Mean Squared Error: ~₹150,000

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Developed as part of Summer Internship Project - Machine Learning for Car Price Prediction

---

⭐ Star this repo if you find it helpful!


