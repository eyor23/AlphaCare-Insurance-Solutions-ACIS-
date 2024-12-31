import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import IncrementalPCA
from sklearn.impute import SimpleImputer
import shap
import lime.lime_tabular

def load_and_clean_data(file_path, delimiter='|', sample_size=0.05):
    """
    Load and clean data from a text file with a specified delimiter, with optional sampling.
    """
    data = pd.read_csv(file_path, delimiter=delimiter, low_memory=False, dtype=str)
    data = data.sample(frac=sample_size, random_state=42)  # Sampling the data
    
    # Handling missing values
    threshold = len(data) * 0.5
    data = data.dropna(thresh=threshold, axis=1)
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    for column in data.select_dtypes(include=['number']).columns:
        data[column] = data[column].fillna(data[column].mean())
    return data

def feature_engineering(data):
    """
    Create new features relevant to TotalPremium and TotalClaims.
    """
    # Convert columns to numeric
    data['TotalClaims'] = pd.to_numeric(data['TotalClaims'], errors='coerce')
    data['TotalPremium'] = pd.to_numeric(data['TotalPremium'], errors='coerce')

    data['ClaimsRatio'] = data['TotalClaims'] / (data['TotalPremium'] + 1e-6)
    return data

def encode_categorical_data(data):
    """
    Convert categorical data into numeric format using one-hot encoding, using pandas Categorical dtype.
    """
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Convert categorical columns to pandas Categorical dtype for better memory efficiency
    for col in categorical_cols:
        data[col] = data[col].astype('category')
    
    encoder = OneHotEncoder(sparse_output=True)
    encoded_data = encoder.fit_transform(data[categorical_cols])
    encoded_data_df = pd.DataFrame.sparse.from_spmatrix(encoded_data)
    encoded_data_df.columns = encoder.get_feature_names_out(categorical_cols)
    
    # Drop original categorical columns and add the one-hot encoded columns
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_data_df], axis=1)
    return data


def apply_incremental_pca(data, n_components=50, batch_size=1000):
    """
    Apply Incremental PCA to reduce dimensionality.
    """
    # Ensure all columns are imputed
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data))
    data_imputed.columns = data.columns
    
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    data_pca = ipca.fit_transform(data_imputed)
    return pd.DataFrame(data_pca)

def train_test_split_data(data, target, test_size=0.3):
    """
    Divide the data into a training set and a test set.
    """
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def build_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    """
    Build and evaluate a model.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} - MSE: {mse}, R2: {r2}")
    return model

def feature_importance(model, X_train):
    """
    Analyze feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        return feature_importance_df.sort_values(by='Importance', ascending=False)
    else:
        print("Feature importance not available for this model.")
        return None

def interpret_model_shap(model, X_train):
    """
    Interpret model predictions using SHAP.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)

def interpret_model_lime(model, X_train, y_train, X_test):
    """
    Interpret model predictions using LIME.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['TotalPremium'], verbose=True, mode='regression')
    for i in range(min(5, len(X_test))):
        exp = explainer.explain_instance(X_test.values[i], model.predict, num_features=10)
        exp.show_in_notebook(show_all=False)
