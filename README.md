# ⚡ Energy Efficiency Prediction System

##  Project Overview
This project leverages **Machine Learning** to predict the energy efficiency of buildings. Specifically, it estimates the **Heating Load (HL)** and **Cooling Load (CL)** required to maintain comfortable indoor air conditions.

By analyzing various building parameters—such as overall height, surface area, and glazing area—this system helps architects and engineers optimize building designs for energy sustainability before construction begins.

## Why This Matters
Energy efficiency is a critical component of modern sustainable architecture.
-   **Sustainability**: Reducing energy consumption lowers the carbon footprint of buildings.
-   **Cost Savings**: Optimized designs lead to significant operational cost reductions over a building's lifecycle.
-   **Speed**: Traditional energy simulation software (like Ecotect) can be computationally expensive. This ML model provides **instant predictions** with high accuracy (R² > 0.97).

## Dataset Information
The dataset was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency).

It contains **768 samples** of different building shapes generated using Ecotect energy simulation software.

### Features (Inputs)
The model takes 8 building characteristics as input:
1.  **X1 - Relative Compactness**: A measure of the building's shape compactness.
2.  **X2 - Surface Area**: Total exterior surface area ($m^2$).
3.  **X3 - Wall Area**: Total wall area ($m^2$).
4.  **X4 - Roof Area**: Total roof area ($m^2$).
5.  **X5 - Overall Height**: Height of the building ($m$).
6.  **X6 - Orientation**: The cardinal direction the building faces.
7.  **X7 - Glazing Area**: The percentage of floor area comprising windows.
8.  **X8 - Glazing Area Distribution**: How the glazing is distributed across the building.

### Targets (Outputs)
1.  **Y1 - Heating Load**: Energy required to warm the building ($kWh/m^2$).
2.  **Y2 - Cooling Load**: Energy required to cool the building ($kWh/m^2$).

## Tech Stack
-   **Python**: Core programming language.
-   **Scikit-Learn**: For training Random Forest and Linear Regression models.
-   **Pandas & NumPy**: Data manipulation and analysis.
-   **Matplotlib & Seaborn**: Data visualization and evaluation plots.
-   **Streamlit**: Interactive web dashboard for real-time predictions.

##  Installation & Usage

### 1. Setup Environment
Ensure you have Python installed (Anaconda recommended).
```bash
pip install -r requirements.txt
```

### 2. Train the Model
You must run the training script first to generate the model files and evaluation plots.



### 3. Run the Web App
Launch the interactive dashboard to make your own predictions.


## Model Performance
We evaluated Linear Regression and Random Forest models. The **Random Forest Regressor** outperformed others with exceptional accuracy:
-   **R² Score**: ~0.98 (Explains 98% of the variance in energy load).
-   **Evaluation**: Check `notebooks/images/model_evaluation_dashboard.png` for a detailed breakdown of residuals and actual vs. predicted values.

---
*Created for the Energy Efficiency ML Project Portfolio.*
