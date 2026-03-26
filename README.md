
# 🏠 House Prices – Advanced Regression Techniques

This repository contains my solution to the Kaggle competition **“House Prices: Advanced Regression Techniques”**, where the goal is to predict residential home prices based on 79 explanatory variables.

## 📊 Project Overview

The objective of this project is to build a regression model that accurately predicts house sale prices using a rich dataset describing various aspects of residential homes in Ames, Iowa.

* **Dataset size:** 79 features
* **Target variable:** `SalePrice`
* **Evaluation metric:** Log Root Mean Squared Error (Log RMSE)
* **Final score:** **0.1983 Log RMSE**

> 🔍 *Log RMSE measures the RMSE between the logarithm of predicted prices and the logarithm of actual sale prices. It reduces the impact of large outliers and focuses on relative differences.*

---

## 🧠 Approach

The project includes the following steps:

* Data cleaning and preprocessing
* Handling missing values
* Feature engineering
* Encoding categorical variables
* Model selection and tuning
* Performance evaluation using Log RMSE

*(You can expand this section later with specific models like XGBoost, Lasso, etc.)*

---

## 📁 Dataset Description

Below is a summarized description of the most important features:

### 🎯 Target Variable

* **SalePrice**: Property sale price in dollars

---

### 🏡 General Property Information

* **MSSubClass**: Building class
* **MSZoning**: General zoning classification
* **LotFrontage**: Linear feet of street connected to property
* **LotArea**: Lot size (sq ft)
* **Street**: Type of road access
* **Alley**: Type of alley access
* **LotShape**: Property shape
* **LandContour**: Property flatness
* **Utilities**: Available utilities
* **LotConfig**: Lot configuration
* **LandSlope**: Property slope

---

### 📍 Location & Conditions

* **Neighborhood**: Physical location within Ames
* **Condition1 / Condition2**: Proximity to roads or railroads

---

### 🏠 Building Characteristics

* **BldgType**: Type of dwelling
* **HouseStyle**: Style of dwelling
* **OverallQual**: Overall material and finish quality
* **OverallCond**: Overall condition
* **YearBuilt**: Construction year
* **YearRemodAdd**: Remodel year

---

### 🏗️ Exterior & Structure

* **RoofStyle / RoofMatl**: Roof type and material
* **Exterior1st / Exterior2nd**: Exterior covering
* **MasVnrType / MasVnrArea**: Masonry veneer type and area
* **ExterQual / ExterCond**: Exterior quality and condition
* **Foundation**: Foundation type

---

### 🧱 Basement Features

* **BsmtQual / BsmtCond**: Basement quality and condition
* **BsmtExposure**: Basement exposure
* **BsmtFinType1 / BsmtFinType2**: Basement finish quality
* **BsmtFinSF1 / BsmtFinSF2**: Finished area (sq ft)
* **BsmtUnfSF**: Unfinished area
* **TotalBsmtSF**: Total basement area

---

### 🔥 Heating & Utilities

* **Heating / HeatingQC**: Heating type and quality
* **CentralAir**: Central air conditioning
* **Electrical**: Electrical system

---

### 🏠 Living Area

* **1stFlrSF / 2ndFlrSF**: Floor areas
* **LowQualFinSF**: Low-quality finished area
* **GrLivArea**: Above-ground living area
* **FullBath / HalfBath**: Bathrooms
* **Bedroom**: Number of bedrooms
* **Kitchen / KitchenQual**: Kitchen count and quality
* **TotRmsAbvGrd**: Total rooms (excluding bathrooms)
* **Functional**: Home functionality

---

### 🚗 Garage

* **GarageType**: Garage location
* **GarageYrBlt**: Year built
* **GarageFinish**: Interior finish
* **GarageCars**: Capacity (cars)
* **GarageArea**: Area (sq ft)
* **GarageQual / GarageCond**: Garage quality and condition

---

### 🌳 Outdoor Features

* **WoodDeckSF**: Deck area
* **OpenPorchSF / EnclosedPorch**: Porch areas
* **3SsnPorch / ScreenPorch**: Seasonal porches
* **PoolArea / PoolQC**: Pool size and quality
* **Fence**: Fence quality
* **MiscFeature / MiscVal**: Miscellaneous features and value

---

## 🚀 Results

* Achieved a **Log RMSE of 0.1983**
* Demonstrates strong predictive performance on a complex, feature-rich dataset

---

## 📌 Future Improvements

* Try advanced ensemble methods (stacking/blending)
* Use feature selection techniques
* Perform deeper hyperparameter tuning
* Explore neural network approaches

---

## 📎 References

* Kaggle Competition: *House Prices - Advanced Regression Techniques*
* Ames Housing Dataset

---

* Turn this into a **top-tier Kaggle-style README (with visuals, pipeline diagram, etc.)**
* Or optimize it for **recruiters / GitHub portfolio impact**
