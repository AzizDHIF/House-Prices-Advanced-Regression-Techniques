url_train = "https://raw.githubusercontent.com/AzizDHIF/House-Prices-Advanced-Regression-Techniques/main/train.csv"
url_test = "https://raw.githubusercontent.com/AzizDHIF/House-Prices-Advanced-Regression-Techniques/main/test.csv"

trainData = pd.read_csv(url_train)
testData = pd.read_csv(url_test)
testData.shape,trainData.shape

testData.head(5)

trainData.head(5)

trainData.columns,testData.columns

features_test=list(testData.columns)
features_train=list(trainData.columns)

print(len(features_test))
print(len(features_train))

x_test=testData
x_train=trainData.drop(columns=['SalePrice'])
y_train=trainData['SalePrice']
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

features_test=list(x_test.columns)
features_train=list(x_train.columns)
print(len(features_test)==len(features_train))

for i in range(len(features_test)):
  if (features_train[i]!=features_test[i]):
    print("il ya un feature qui n'est pas le même")

x_train.describe()

x_train.info()

x_test.info()

x_test=x_test.drop(columns=['MasVnrType','FireplaceQu','MiscFeature'])
x_train=x_train.drop(columns=['MasVnrType','FireplaceQu','MiscFeature'])

colonneNotNull=['Alley','PoolQC','Fence']

# Remplacer les valeurs NaN par 'None' pour les colonnes concernées
for col in colonneNotNull:
    x_train[col] = x_train[col].fillna("None").str.strip()
    x_test[col] = x_test[col].fillna("None").str.strip()

# Encodage ordinal pour PoolQC (qualité piscine)
pool_qc_mapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0}  # 'None' = Pas de piscine
x_train['PoolQC'] = x_train['PoolQC'].map(pool_qc_mapping)
x_test['PoolQC'] = x_test['PoolQC'].map(pool_qc_mapping)

# Encodage one-hot pour les autres variables catégorielles
x_train = pd.get_dummies(x_train, columns=['Alley', 'Fence'], drop_first=True)
x_test = pd.get_dummies(x_test, columns=['Alley', 'Fence'], drop_first=True)


# Vérification des modifications
x_train.head()
x_test.head()

y_train.info()

x_train.info()

# on garde les id de x_test pour le submission csv , mais on peut enlever ceux du x_train

x_train=x_train.drop(columns=['Id'])
x_test_id=x_test['Id']
x_test=x_test.drop(columns=['Id'])
x_test

#traitement des features categorielle
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Identifier les colonnes numériques et catégorielles


categorical_nominal = [
    'MSZoning', 'Street',  'LotConfig', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation',
    'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive',
     'SaleType', 'SaleCondition','MSSubClass', 'Utilities'
]

categorical_ordinal = [
    'LotShape', 'LandContour', 'LandSlope', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'Functional',  'GarageFinish',
    'GarageQual', 'GarageCond'
]

numeric = [
     'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    'MoSold', 'YrSold'
]


features=categorical_nominal+categorical_ordinal+numeric
print(len(features))

import numpy as np
import pandas as pd
from scipy import stats

# Fusion des données
data = x_train.copy()
data['SalePrice'] = y_train

# Liste des colonnes numériques
numeric = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    'MoSold', 'YrSold'
]

from collections import defaultdict
from scipy import stats
import numpy as np

# Dictionnaire pour stocker les indices et leur z-score extrême
index_zscore_max = defaultdict(float)

# Calcul des z-scores pour chaque (feature, SalePrice)
for feature in numeric:
    subset = data[[feature, 'SalePrice']].dropna()
    z = np.abs(stats.zscore(subset))  # shape (n, 2)
    max_z = np.max(z, axis=1)         # score max entre feature et SalePrice

    for idx, score in zip(subset.index, max_z):
        index_zscore_max[idx] = max(index_zscore_max[idx], score)

# Garder les 30 lignes les plus extrêmes
sorted_outliers = sorted(index_zscore_max.items(), key=lambda x: x[1], reverse=True)
top_outliers = [idx for idx, _ in sorted_outliers[:6]]

print(f"Suppression des {len(top_outliers)} outliers les plus extrêmes liés aux features numériques et à SalePrice")

# Supprimer dans les jeux d'entraînement
x_train = x_train.drop(index=top_outliers)
y_train = y_train.drop(index=top_outliers)

# Réindexer proprement
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

#Descritpiion des données
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def describe_data(data):
    description = data.describe(include='all')
    return description


data_description = describe_data(x_train)
print(data_description)

#Corrélation avec la variable cible

# Corrélation
corr = trainData.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
print(corr.head(10))  # top corrélées
print(corr.tail(5))   # inversément corrélées

# Heatmap
plt.figure(figsize=(10,8))
top_corr = trainData.corr(numeric_only=True).nlargest(10, 'SalePrice')['SalePrice'].index
sns.heatmap(trainData[top_corr].corr(), annot=True, cmap='coolwarm')
plt.title("Top 10 corrélations avec SalePrice")
plt.show()

#visualisation des correlation > 0,7
# Variables très corrélées
threshold = 0.7
corr_matrix = trainData.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = [column for column in upper.columns if any(upper[column] > threshold)]
print("Variables très corrélées entre elles (corr > 0.7):", high_corr)

# Visualiser la distribution des données des valeur numeric
x_train[numeric].hist(bins=30, figsize=(15, 10))
plt.suptitle('Feature Distribution')
plt.show()

#test D'anova ( le quartier influence il le prix de vente )

import scipy.stats as stats

groups = [y["SalePrice"].values for x, y in trainData.groupby("Neighborhood")]
anova_result = stats.f_oneway(*groups)
print(f"p-value ANOVA pour Neighborhood ~ SalePrice: {anova_result.pvalue:.4f}")

# encoding categorical ordinal features

ordinal_mappings = {
    'LotShape': {
        'Reg': 3,
        'IR1': 2,
        'IR2': 1,
        'IR3': 0

    },

    'LandContour': {
        'Lvl': 3,
        'Bnk': 2,
        'HLS': 1,
        'Low': 0

    },
    'LandSlope': {
        'Gtl': 2,
        'Mod': 1,
        'Sev': 0

    },
    'ExterQual': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },
    'ExterCond': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },

    'BsmtQual': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },
    'BsmtCond': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },
    'BsmtExposure': {
        'Gd': 3,
        'Av': 2,
        'Mn': 1,
        'No': 0

    },
    'BsmtFinType1': {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1

    },
    'BsmtFinType2': {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1

    },
    'HeatingQC': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },
    'KitchenQual': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    },
    'Functional': {
        'Typ': 7,
        'Min1': 6,
        'Min2': 5,
        'Mod': 4,
        'Maj1': 3,
        'Maj2': 2,
        'Sev': 1,
        'Sal': 0

    },
    'GarageFinish': {
        'Fin': 3,
        'RFn': 2,
        'Unf': 1

    },
    'GarageQual': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0,

    },
    'GarageCond': {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'Po': 0

    }
}

# Function to strip spaces and map ordinal values
def clean_and_map_ordinal_values(df, mappings):
  for feature, mapping in mappings.items():
    df[feature] = df[feature].apply(lambda x: mapping[str(x).strip()] if str(x).strip() in mapping else x)
  return df

# Apply the function to x_train
x_train = clean_and_map_ordinal_values(x_train, ordinal_mappings)
x_train.info()

x_train[categorical_ordinal]

x_test = clean_and_map_ordinal_values(x_test, ordinal_mappings)
x_test.info()
x_test[categorical_ordinal]

x_test.info()

# Imputer pour les features nominales (valeur la plus fréquente)
imputer_nominal = SimpleImputer(strategy='most_frequent')
x_train[categorical_nominal] = imputer_nominal.fit_transform(x_train[categorical_nominal])
x_test[categorical_nominal] = imputer_nominal.transform(x_test[categorical_nominal])

# Imputer pour les features ordinales (médiane)
imputer_ordinal = SimpleImputer(strategy='median')
x_train[categorical_ordinal] = imputer_ordinal.fit_transform(x_train[categorical_ordinal])
x_test[categorical_ordinal] = imputer_ordinal.transform(x_test[categorical_ordinal])

# Imputer pour les features numériques (moyenne)
imputer_numeric = SimpleImputer(strategy='median')
x_train[numeric] = imputer_numeric.fit_transform(x_train[numeric])
x_test[numeric] = imputer_numeric.transform(x_test[numeric])

x_train.info(),x_test.info()

from sklearn.preprocessing import OneHotEncoder


# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the encoder on x_train
encoded_train = encoder.fit_transform(x_train[categorical_nominal])

# Transform x_test using the fitted encoder
encoded_test = encoder.transform(x_test[categorical_nominal])

# Convert the encoded arrays back to DataFrames
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_nominal))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_nominal))

# Drop the original categorical columns from x_train and x_test
x_train = x_train.drop(columns=categorical_nominal)
x_test = x_test.drop(columns=categorical_nominal)

# Concatenate the encoded columns with the rest of the DataFrame
x_train = pd.concat([x_train, encoded_train_df], axis=1)
x_test = pd.concat([x_test, encoded_test_df], axis=1)

x_test

#traitement de quelques features numériques
x_train['TotalSF'] = x_train['TotalBsmtSF'] + x_train['1stFlrSF'] + x_train['2ndFlrSF']
x_test['TotalSF'] = x_test['TotalBsmtSF'] + x_test['1stFlrSF'] + x_test['2ndFlrSF']

x_train['TotalBathrooms'] = x_train['FullBath'] + 0.5 * x_train['HalfBath'] + x_train['BsmtFullBath'] + 0.5 * x_train['BsmtHalfBath']
x_test['TotalBathrooms'] = x_test['FullBath'] + 0.5 * x_test['HalfBath'] + x_test['BsmtFullBath'] + 0.5 * x_test['BsmtHalfBath']

x_train['HouseAge'] = 2020 - x_train['YearBuilt']
x_test['HouseAge'] = 2020 - x_test['YearBuilt']

x_train['SinceRemod'] = 2020 - x_train['YearRemodAdd']
x_test['SinceRemod'] = 2020 - x_test['YearRemodAdd']

x_train['HasGarage'] = (x_train['GarageArea'] > 0).astype(int)
x_test['HasGarage'] = (x_test['GarageArea'] > 0).astype(int)

x_train

!pip install xgboost

from sklearn.preprocessing import RobustScaler

# --- Initialisation du scaler
scaler = RobustScaler()

# --- Application du scaler uniquement sur les colonnes numériques
x_train[numeric] = scaler.fit_transform(x_train[numeric])
x_test[numeric] = scaler.transform(x_test[numeric])  # Utilisation de transform() au lieu de fit_transform() sur les données de test
y_train

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Colonnes numériques hors SalePrice
X = trainData.select_dtypes(include='number').drop(columns='SalePrice', errors='ignore')
y = trainData['SalePrice']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualisation
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=20)
plt.colorbar(scatter, label='SalePrice')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('PCA (NaN remplacés par moyenne, données standardisées)')
plt.grid(True)
plt.show()

import numpy as np

# --- Application du logarithme naturel sur la colonne 'price'

y_train = y_train = np.log1p(y_train)
y_train

!pip install xgboost

!pip install lightgbm

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
lasso = Lasso(alpha=np.float64(0.000379269019073225), random_state=42)

xgb = XGBRegressor(
    colsample_bytree=0.6,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    reg_alpha=0.01,
    reg_lambda=5,
    subsample=0.6,
    random_state=42
)
cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_state=42
)
ridge = Ridge(alpha=10, random_state=42)

lgbm = LGBMRegressor(
    colsample_bytree=0.6,
    learning_rate=0.05,
    max_depth=5,
    min_child_samples=5,
    n_estimators=300,
    num_leaves=15,
    reg_alpha=0.01,
    reg_lambda=0.5,
    subsample=0.6,
    random_state=42
)
gbr = GradientBoostingRegressor(
    n_estimators=3000, learning_rate=0.05,
    max_depth=4, max_features='sqrt',
    min_samples_leaf=15, min_samples_split=10,
    loss='huber', random_state=42
)
# Définir le modèle de stacking
stacked_model = StackingRegressor(
    estimators=[
        ('gbr', gbr),
        ('lasso', lasso),
        ('xgb', xgb),
        ('lgbm', lgbm)

    ],
    final_estimator=Lasso(alpha=0.001),  # Par exemple, Lasso comme méta-modèle
    passthrough=True,  # Pour passer aussi les features d’origine au méta-modèle
    n_jobs=-1
)

# Entraînement du modèle de stacking
stacked_model.fit(x_train, y_train)

# Évaluation
y_pred = stacked_model.predict(x_test)

y_pred=np.expm1(y_pred)

import pandas as pd
import os

df = pd.DataFrame({'Id': x_test_id,'SalePrice': y_pred
})
df.to_csv("predictionStacking.csv", index=False)
print("Fichier 'predictionStacking.csv' créé dans le dossier courant.")

from google.colab import files
files.download('predictionStacking.csv')