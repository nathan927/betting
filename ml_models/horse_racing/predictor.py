# ml_models/horse-racing/enhanced_predictor.py
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

# Specific model imports
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Flask for API (optional, can be a separate service)
from flask import Flask, request, jsonify
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Feature Engineering ---
class AdvancedFeatureEngineer:
    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.ct = None # ColumnTransformer
        self.fitted_columns_ = None # Store all feature names after fitting
        self.original_numerical_features = numerical_features[:] # Keep a copy
        self.original_categorical_features = categorical_features[:] # Keep a copy


    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example interactions (domain-specific)
        if 'jockey_win_rate' in df.columns and 'trainer_win_rate' in df.columns:
            df['jockey_trainer_interaction'] = df['jockey_win_rate'] * df['trainer_win_rate']
        if 'horse_age' in df.columns and 'horse_weight' in df.columns:
            df['age_weight_ratio'] = df['horse_age'] / (df['horse_weight'] + 1e-6)
        # Add more based on EDA
        return df

    def _create_lagged_features(self, df: pd.DataFrame, group_by_col: str, target_col: str, lags: List[int]) -> pd.DataFrame:
        df_sorted = df.sort_values(by=[group_by_col, 'race_date'])
        for lag in lags:
            lagged_col_name = f'{target_col}_lag_{lag}'
            df[lagged_col_name] = df_sorted.groupby(group_by_col)[target_col].shift(lag)
        return df

    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        logger.info("Starting advanced feature engineering...")
        df_processed = df.copy()

        # Impute missing values first
        for col in self.original_numerical_features: # Use original list for imputation
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col] = SimpleImputer(strategy='median').fit_transform(df_processed[[col]])
        for col in self.original_categorical_features: # Use original list for imputation
             if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col] = SimpleImputer(strategy='most_frequent').fit_transform(df_processed[[col]]).ravel()

        df_processed = self._create_interaction_features(df_processed)

        # Example lagged features (ensure 'race_date' and 'horse_id' exist)
        if 'speed_figure' in df_processed.columns and 'horse_id' in df_processed.columns and 'race_date' in df_processed.columns:
            df_processed = self._create_lagged_features(df_processed, 'horse_id', 'speed_figure', [1, 2, 3])

        # Dynamically identify all numerical and categorical columns present after feature creation
        current_numerical_features = []
        current_categorical_features = []

        for col in df_processed.columns:
            if col in ['race_id', 'horse_id', 'win', 'place', 'race_date', 'horse_name']: # Exclude IDs, targets, date
                continue
            if df_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_processed[col]):
                current_categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(df_processed[col]):
                current_numerical_features.append(col)

        # Ensure original features are still considered if they exist
        final_numerical_features = sorted(list(set(self.original_numerical_features + current_numerical_features) & set(df_processed.columns)))
        final_categorical_features = sorted(list(set(self.original_categorical_features + current_categorical_features) & set(df_processed.columns)))


        if is_training or self.ct is None:
            numerical_transformer = Pipeline(steps=[
                ('imputer_num', SimpleImputer(strategy='median')), # Impute again just in case new features have NaNs
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer_cat', SimpleImputer(strategy='most_frequent')), # Impute again
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            self.ct = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, final_numerical_features),
                    ('cat', categorical_transformer, final_categorical_features)
                ],
                remainder='drop'
            )
            logger.info("Fitting ColumnTransformer...")
            transformed_data = self.ct.fit_transform(df_processed)
            self.fitted_columns_ = self._get_fitted_columns(final_numerical_features, final_categorical_features)
        else:
            logger.info("Transforming data using existing ColumnTransformer...")
            # Ensure columns are in the same order as during fitting
            # df_reordered = df_processed[self.ct.feature_names_in_] if hasattr(self.ct, 'feature_names_in_') else df_processed
            transformed_data = self.ct.transform(df_processed) # Relies on ColumnTransformer to handle column order/presence

        logger.info(f"Feature engineering complete. Shape of transformed data: {transformed_data.shape}")
        if transformed_data.shape[1] == 0:
            logger.warning("No features were generated by the feature engineering process. Check feature lists and data.")
        return transformed_data

    def _get_fitted_columns(self, final_num_features: List[str], final_cat_features: List[str]) -> List[str]:
        if not self.ct: return []

        num_feature_names = final_num_features
        cat_feature_names_out = []
        try:
            cat_transformer = self.ct.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names_out = list(cat_transformer.get_feature_names_out(final_cat_features))
        except Exception as e:
            logger.warning(f"Could not get OHE feature names: {e}")

        return num_feature_names + cat_feature_names_out

    def get_feature_names(self) -> List[str]:
        return self.fitted_columns_ if self.fitted_columns_ is not None else []


# --- Neural Network Model (PyTorch) ---
class RacingNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.3):
        super(RacingNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1)) # Output for binary classification (win)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- Ensemble Predictor ---
class EnsembleHorseRacingPredictor:
    def __init__(self, models_config: Optional[Dict[str, Any]] = None, feature_engineer_config: Optional[Dict[str,List[str]]] = None):
        if feature_engineer_config is None: # Define default feature columns here
            feature_engineer_config = {
                'categorical_features': ['track_condition', 'race_class', 'jockey_id', 'trainer_id', 'venue_id', 'horse_sex'],
                'numerical_features': ['horse_age', 'horse_weight', 'draw', 'speed_figure_avg', 'days_since_last_race',
                                       'weight_carried_diff', 'recent_win_rate', 'recent_avg_finish_pos']
            }
        self.feature_engineer = AdvancedFeatureEngineer(**feature_engineer_config)

        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        if models_config is None: models_config = self._get_default_models_config()
        self._init_models(models_config)

    def _get_default_models_config(self) -> Dict[str, Any]:
        return {
            'xgb': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8),
            'lgb': lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, max_depth=7, subsample=0.8, colsample_bytree=0.8),
            'rf': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=150, max_depth=12, min_samples_split=10, min_samples_leaf=5),
            'nn': None,
            'logreg_meta': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=0.1)
        }

    def _init_models(self, config: Dict[str, Any]):
        self.models = {}
        for name, model_instance in config.items():
            if name == 'nn' and model_instance is None: continue
            self.models[name] = model_instance
        logger.info(f"Initialized models: {list(self.models.keys())}")

    def train_all_models(self, df: pd.DataFrame, target_col: str = 'win'):
        logger.info("Starting training for all models...")
        y = df[target_col].astype(int)
        # Exclude target and non-feature columns before feature engineering
        feature_df = df.drop(columns=[target_col, 'race_id', 'horse_id', 'place', 'horse_name', 'race_date'], errors='ignore')
        X_transformed = self.feature_engineer.fit_transform(feature_df, is_training=True)

        if X_transformed.shape[1] == 0:
            logger.error("No features were produced by feature engineering. Aborting training.")
            self.is_trained = False
            return

        if 'nn' not in self.models or self.models['nn'] is None:
            input_dim = X_transformed.shape[1]
            self.models['nn'] = RacingNN(input_dim=input_dim).to(self.device)
            logger.info(f"NN model initialized with input_dim: {input_dim}")

        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_model_predictions_cv = np.zeros((X_transformed.shape[0], len(self.models) -1)) # -1 for meta learner

        for i, (name, model) in enumerate(filter(lambda item: item[0] != 'logreg_meta', self.models.items())):
            logger.info(f"Training model: {name}...")
            if name == 'nn':
                self._train_nn(X_transformed, y.values, epochs=100, batch_size=512) # Adjust params
                self.models['nn'].eval()
                with torch.no_grad():
                    tensor_x = torch.FloatTensor(X_transformed).to(self.device)
                    logits = self.models['nn'](tensor_x)
                    base_model_predictions_cv[:, i] = torch.sigmoid(logits).cpu().numpy().flatten()
            else:
                sample_weights = compute_sample_weight('balanced', y)
                # Note: Some models like XGBoost have built-in ways to handle class imbalance (e.g. scale_pos_weight)
                model.fit(X_transformed, y, sample_weight=sample_weights if name in ['rf', 'lgb'] else None)
                base_model_predictions_cv[:, i] = model.predict_proba(X_transformed)[:, 1]

        logger.info("Training meta-learner (Logistic Regression)...")
        self.models['logreg_meta'].fit(base_model_predictions_cv, y)

        self.is_trained = True
        logger.info("All models trained successfully.")

    def _train_nn(self, X_train_transformed: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int):
        model_nn = self.models['nn']
        model_nn.train()
        X_tensor = torch.FloatTensor(X_train_transformed).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(model_nn.parameters(), lr=0.001, weight_decay=1e-4)
        # Calculate pos_weight for BCEWithLogitsLoss
        pos_weight_val = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).float().to(self.device))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.2, verbose=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model_nn(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"NN Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained: raise RuntimeError("Models are not trained yet.")

        logger.info("Making predictions with ensemble...")
        # Ensure feature columns used for prediction match those used for training (excluding target, etc.)
        feature_df = df.drop(columns=['win', 'race_id', 'horse_id', 'place', 'horse_name', 'race_date'], errors='ignore')
        X_transformed = self.feature_engineer.fit_transform(feature_df, is_training=False)

        if X_transformed.shape[1] == 0:
            logger.warning("No features produced for prediction. Returning empty array.")
            return np.array([])

        base_model_predictions = np.zeros((X_transformed.shape[0], len(self.models) -1))
        for i, (name, model) in enumerate(filter(lambda item: item[0] != 'logreg_meta', self.models.items())):
            if name == 'nn':
                model.eval()
                with torch.no_grad():
                    tensor_x = torch.FloatTensor(X_transformed).to(self.device)
                    logits = model(tensor_x)
                    base_model_predictions[:, i] = torch.sigmoid(logits).cpu().numpy().flatten()
            else:
                base_model_predictions[:, i] = model.predict_proba(X_transformed)[:, 1]

        final_predictions = self.models['logreg_meta'].predict_proba(base_model_predictions)[:, 1]
        logger.info(f"Generated {len(final_predictions)} predictions.")
        return final_predictions

    def save_model(self, path: str = 'ml_models/horse_racing/predictor_ensemble.joblib'):
        logger.info(f"Saving model ensemble to {path}")
        model_pack = {
            'feature_engineer': self.feature_engineer,
            'models': {name: model for name, model in self.models.items() if name != 'nn'},
            'nn_state_dict': self.models['nn'].state_dict() if 'nn' in self.models and self.models['nn'] is not None else None,
            'nn_input_dim': self.models['nn'].network[0].in_features if ('nn' in self.models and self.models['nn'] is not None and hasattr(self.models['nn'],'network')) else None,
            'is_trained': self.is_trained,
            'fitted_columns': self.feature_engineer.fitted_columns_ # Save names of features model was trained on
        }
        joblib.dump(model_pack, path)
        logger.info("Model ensemble saved.")

    @classmethod
    def load_model(cls, path: str = 'ml_models/horse_racing/predictor_ensemble.joblib'):
        logger.info(f"Loading model ensemble from {path}")
        model_pack = joblib.load(path)

        # Create instance, feature_engineer_config will be overwritten by loaded object
        predictor = cls(feature_engineer_config=None)
        predictor.feature_engineer = model_pack['feature_engineer']
        predictor.feature_engineer.fitted_columns_ = model_pack.get('fitted_columns') # Load feature names

        loaded_models = model_pack['models']
        if model_pack.get('nn_state_dict') and model_pack.get('nn_input_dim'):
            nn_model = RacingNN(input_dim=model_pack['nn_input_dim'])
            nn_model.load_state_dict(model_pack['nn_state_dict'])
            nn_model.to(predictor.device); nn_model.eval()
            loaded_models['nn'] = nn_model

        predictor.models = loaded_models
        predictor.is_trained = model_pack['is_trained']
        logger.info("Model ensemble loaded successfully.")
        return predictor

app = Flask(__name__)
predictor_instance: Optional[EnsembleHorseRacingPredictor] = None

def initialize_predictor_api():
    global predictor_instance
    try:
        model_path = 'ml_models/horse_racing/predictor_ensemble.joblib'
        predictor_instance = EnsembleHorseRacingPredictor.load_model(model_path)
    except Exception as e:
        logger.error(f"API: Error loading predictor: {e}")

@app.route('/predict/horse-racing', methods=['POST'])
def predict_horse_racing_api():
    if not predictor_instance or not predictor_instance.is_trained:
        return jsonify({"error": "Model not ready or not trained"}), 503
    data = request.get_json()
    if not data or 'races' not in data: return jsonify({"error": "Missing 'races' data"}), 400
    try:
        race_df = pd.DataFrame(data['races'])
        probabilities = predictor_instance.predict_proba(race_df)
        # Match results back to input entries, assuming input has unique IDs
        results = [{'input_data': data['races'][i], 'win_probability': float(p)} for i,p in enumerate(probabilities)]
        return jsonify({"predictions": results}), 200
    except Exception as e: return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/status/horse-racing', methods=['GET'])
def status_horse_racing_api():
    if predictor_instance:
        return jsonify({
            "model_loaded": True,
            "is_trained": predictor_instance.is_trained,
            "feature_names_count": len(predictor_instance.feature_engineer.get_feature_names()) if predictor_instance.feature_engineer.get_feature_names() else 0
        }), 200
    return jsonify({"model_loaded": False}), 503

if __name__ == '__main__':
    logger.info("Horse Racing Enhanced Predictor Service - Main Block")
    # For training, you would typically run a separate script that loads data,
    # initializes EnsembleHorseRacingPredictor, calls train_all_models, and saves the model.
    # Example:
    # print("To train model, uncomment and adapt training data loading section in a dedicated script.")
    # print("Starting Flask API for serving predictions (if model exists).")

    initialize_predictor_api() # Load model at startup
    app.run(host='0.0.0.0', port=5002, debug=False)
```
