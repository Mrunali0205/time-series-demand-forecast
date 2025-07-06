# Time Series Demand Forecast: ARMA & Prophet Models

## 📊 Project Overview

This project implements advanced time series forecasting techniques to predict demand using multiple models including ARMA (AutoRegressive Moving Average) and Prophet. The analysis is based on the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/overview) dataset from Kaggle.

## 🎯 Objectives

- **Demand Prediction**: Forecast future demand using historical time series data
- **Model Comparison**: Evaluate different forecasting approaches (ARMA, Prophet, Neural Prophet)
- **Feature Engineering**: Create seasonal and trend-based features
- **Performance Analysis**: Compare model accuracy using Mean Absolute Error (MAE)

## 📁 Project Structure

```
time-series-demand-forecast/
├── time-series-demand-forecast-arma-prophet copy.ipynb  # Main analysis notebook
└── README.md                                           # This file
```

## 🔧 Technologies & Libraries Used

### Core Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities

### Time Series Analysis
- **Statsmodels** - Statistical models including ARMA
- **PMDARIMA** - Auto ARIMA implementation
- **Prophet** - Facebook's forecasting tool
- **Neural Prophet** - Neural network-based forecasting

### Additional Tools
- **Torch** - Deep learning framework (for Neural Prophet)
- **Yellowbrick** - Model selection visualization

## 📈 Models Implemented

### 1. ARMA (AutoRegressive Moving Average)
- **Purpose**: Traditional statistical time series forecasting
- **Features**: 
  - Manual differencing for stationarity
  - Fourier features for seasonality
  - ADF test for stationarity validation
- **Use Case**: Baseline statistical approach

### 2. Prophet
- **Purpose**: Facebook's automated forecasting tool
- **Features**:
  - Automatic seasonality detection
  - Trend modeling
  - Holiday effects
  - Changepoint detection
- **Use Case**: Automated forecasting with minimal parameter tuning

### 3. Neural Prophet
- **Purpose**: Neural network-based forecasting
- **Features**:
  - Deep learning approach
  - Automatic hyperparameter optimization
  - Multiple seasonality patterns
- **Use Case**: Advanced forecasting with neural networks

### 4. Additional Models
- **Exponential Smoothing** (Holt-Winters)
- **Random Forest Regression**
- **Decision Tree Regression**
- **Support Vector Regression**

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn
pip install statsmodels pmdarima
pip install prophet neuralprophet
pip install scikit-learn yellowbrick
pip install torch
```

### Running the Analysis
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mrunali0205/time-series-demand-forecast.git
   cd time-series-demand-forecast
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook "time-series-demand-forecast-arma-prophet copy.ipynb"
   ```

3. **Run all cells** to reproduce the analysis

## 📊 Dataset Information

The project uses the **Store Item Demand Forecasting Challenge** dataset which contains:
- **Time Series Data**: Historical demand data
- **Features**: Date, store, item information
- **Target Variable**: Demand/sales quantity
- **Seasonality**: Clear seasonal patterns in the data

## 🔍 Key Analysis Steps

### 1. Data Exploration
- Dataset overview and statistics
- Time series visualization
- Seasonality and trend analysis
- Stationarity testing (ADF test)

### 2. Feature Engineering
- Date-based features (day of week, month, year)
- Seasonal indicators
- Trend features
- Lag features for time series

### 3. Model Development
- **ARMA**: Manual differencing and Fourier features
- **Prophet**: Automatic seasonality and trend modeling
- **Neural Prophet**: Deep learning approach
- **Ensemble**: Combining multiple models

### 4. Model Evaluation
- **Metrics**: Mean Absolute Error (MAE)
- **Cross-validation**: Time series cross-validation
- **Visualization**: Forecast plots and residual analysis

## 📈 Results & Insights

### Model Performance
- **Prophet**: Generally performs well with automatic seasonality detection
- **Neural Prophet**: Advanced capabilities for complex patterns
- **ARMA**: Good baseline for stationary time series
- **Ensemble**: Often provides the best overall performance

### Key Findings
- Clear seasonal patterns in demand data
- Trend components present in the time series
- Multiple models provide complementary strengths
- Feature engineering significantly improves performance

## 🛠️ Customization

### Adding New Models
1. Import required libraries
2. Prepare data in the required format
3. Train the model
4. Evaluate using MAE metric
5. Add to comparison analysis

### Parameter Tuning
- **Prophet**: Adjust seasonality modes and changepoint priors
- **Neural Prophet**: Modify epochs, batch size, and learning rate
- **ARMA**: Optimize p, d, q parameters

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the dataset
- **Facebook Research** for Prophet
- **Statsmodels** team for statistical tools
- **Scikit-learn** team for machine learning utilities

## 📞 Contact

- **GitHub**: [Mrunali0205](https://github.com/Mrunali0205)
- **Project Link**: [https://github.com/Mrunali0205/time-series-demand-forecast](https://github.com/Mrunali0205/time-series-demand-forecast)

---

**Note**: This project serves as a comprehensive guide for time series forecasting, demonstrating multiple approaches from traditional statistical methods to modern deep learning techniques. 