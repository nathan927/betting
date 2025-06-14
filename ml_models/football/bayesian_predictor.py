# ml_models/football/bayesian_predictor.py
import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import redis
from flask import Flask, request, jsonify
import threading
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask App for serving predictions
app = Flask(__name__)

# Redis connection (replace with your actual Redis config)
try:
    redis_client = redis.StrictRedis(host=process.env.REDIS_HOST || 'localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except Exception as e:
    logger.error(f"Could not connect to Redis: {e}")
    redis_client = None

class BayesianFootballPredictor:
    def __init__(self, samples=5000, tune=2000):
        self.samples = samples
        self.tune = tune
        self.model = None
        self.trace = None
        self.team_id_map = {}
        self.id_team_map = {}
        self.trained_teams = 0
        self.last_trained_date = None

    def _preprocess_data(self, historical_data_df):
        """
        Preprocesses historical match data.
        Expects a DataFrame with columns: ['home_team', 'away_team', 'home_score', 'away_score']
        """
        teams = pd.concat([historical_data_df['home_team'], historical_data_df['away_team']]).unique()
        teams.sort()
        self.team_id_map = {team: i for i, team in enumerate(teams)}
        self.id_team_map = {i: team for team, i in self.team_id_map.items()}
        self.trained_teams = len(teams)

        historical_data_df['home_team_id'] = historical_data_df['home_team'].map(self.team_id_map)
        historical_data_df['away_team_id'] = historical_data_df['away_team'].map(self.team_id_map)

        home_goals = historical_data_df['home_score'].values.astype(int)
        away_goals = historical_data_df['away_score'].values.astype(int)
        home_team_idx = historical_data_df['home_team_id'].values.astype(int)
        away_team_idx = historical_data_df['away_team_id'].values.astype(int)

        return home_goals, away_goals, home_team_idx, away_team_idx

    def build_model(self, home_goals, away_goals, home_team_idx, away_team_idx):
        """
        Builds the Bayesian hierarchical model using PyMC3.
        """
        logger.info(f"Building model with {self.trained_teams} teams.")
        with pm.Model() as model:
            # Hyperpriors for team parameters
            home_advantage_mu = pm.Normal('home_advantage_mu', mu=0.5, sigma=0.5)
            attack_mu = pm.Normal('attack_mu', mu=0, sigma=1)
            defence_mu = pm.Normal('defence_mu', mu=0, sigma=1)

            attack_sigma = pm.HalfNormal('attack_sigma', sigma=0.5)
            defence_sigma = pm.HalfNormal('defence_sigma', sigma=0.5)

            # Team-specific parameters
            home_advantage = pm.Normal('home_advantage', mu=home_advantage_mu, sigma=0.2, shape=self.trained_teams) # Home advantage for each team
            attack_strength = pm.Normal('attack_strength', mu=attack_mu, sigma=attack_sigma, shape=self.trained_teams)
            defence_strength = pm.Normal('defence_strength', mu=defence_mu, sigma=defence_sigma, shape=self.trained_teams)

            # Intercept (average goal rate)
            intercept = pm.Normal('intercept', mu=0.7, sigma=0.5) # Based on typical goals per match

            # Expected number of goals
            home_theta = tt.exp(intercept + home_advantage[home_team_idx] + attack_strength[home_team_idx] - defence_strength[away_team_idx])
            away_theta = tt.exp(intercept + attack_strength[away_team_idx] - defence_strength[home_team_idx])

            # Likelihood of observed goals (Poisson distribution)
            pm.Poisson('home_goals_obs', mu=home_theta, observed=home_goals)
            pm.Poisson('away_goals_obs', mu=away_theta, observed=away_goals)

        self.model = model
        logger.info("Model build complete.")

    def train(self, historical_data_df):
        """
        Trains the model with historical data.
        """
        if self.model is None:
            home_goals, away_goals, home_team_idx, away_team_idx = self._preprocess_data(historical_data_df)
            self.build_model(home_goals, away_goals, home_team_idx, away_team_idx)

        logger.info(f"Starting MCMC sampling. Samples: {self.samples}, Tune: {self.tune}")
        with self.model:
            # Using NUTS sampler
            self.trace = pm.sample(self.samples, tune=self.tune, cores=1, return_inferencedata=True, target_accept=0.9) # target_accept for better sampling

        self.last_trained_date = datetime.utcnow()
        logger.info("MCMC sampling complete. Model trained.")
        self._store_model_summary_to_redis()

    def _store_model_summary_to_redis(self):
        if not redis_client or self.trace is None:
            logger.warning("Redis client not available or model not trained. Skipping storing summary.")
            return
        try:
            summary = az.summary(self.trace, var_names=['intercept', 'home_advantage_mu', 'attack_mu', 'defence_mu'])
            redis_client.set("football_model:summary", summary.to_json())
            redis_client.set("football_model:last_trained", self.last_trained_date.isoformat())
            redis_client.set("football_model:trained_teams_map", json.dumps(self.team_id_map))
            logger.info("Model summary and metadata stored in Redis.")
        except Exception as e:
            logger.error(f"Error storing model summary to Redis: {e}")


    def predict_match_outcome_probs(self, home_team_name, away_team_name, n_simulations=10000):
        """
        Predicts outcome probabilities (Home Win, Draw, Away Win) and common scores.
        """
        if self.trace is None:
            raise ValueError("Model has not been trained yet.")
        if home_team_name not in self.team_id_map or away_team_name not in self.team_id_map:
            raise ValueError("One or both teams not found in trained model.")

        home_id = self.team_id_map[home_team_name]
        away_id = self.team_id_map[away_team_name]

        with self.model: # Context of the original model
            # Simulate games using posterior samples
            ppc_home_goals = []
            ppc_away_goals = []

            for i in np.random.randint(0, len(self.trace.posterior.chain) * len(self.trace.posterior.draw), n_simulations):
                chain_idx = i // len(self.trace.posterior.draw)
                draw_idx = i % len(self.trace.posterior.draw)

                intercept_sample = self.trace.posterior['intercept'][chain_idx, draw_idx].values
                home_adv_sample = self.trace.posterior['home_advantage'][chain_idx, draw_idx, home_id].values
                home_att_sample = self.trace.posterior['attack_strength'][chain_idx, draw_idx, home_id].values
                home_def_sample = self.trace.posterior['defence_strength'][chain_idx, draw_idx, home_id].values

                away_att_sample = self.trace.posterior['attack_strength'][chain_idx, draw_idx, away_id].values
                away_def_sample = self.trace.posterior['defence_strength'][chain_idx, draw_idx, away_id].values

                home_theta_sample = np.exp(intercept_sample + home_adv_sample + home_att_sample - away_def_sample)
                away_theta_sample = np.exp(intercept_sample + away_att_sample - home_def_sample)

                ppc_home_goals.append(np.random.poisson(home_theta_sample))
                ppc_away_goals.append(np.random.poisson(away_theta_sample))

        ppc_home_goals = np.array(ppc_home_goals)
        ppc_away_goals = np.array(ppc_away_goals)

        # Calculate outcome probabilities
        home_wins = np.sum(ppc_home_goals > ppc_away_goals) / n_simulations
        away_wins = np.sum(ppc_away_goals > ppc_home_goals) / n_simulations
        draws = np.sum(ppc_home_goals == ppc_away_goals) / n_simulations

        # Most common scores
        scores_df = pd.DataFrame({'home_goals': ppc_home_goals, 'away_goals': ppc_away_goals})
        common_scores = scores_df.groupby(['home_goals', 'away_goals']).size().nlargest(5).reset_index(name='count')
        common_scores['probability'] = common_scores['count'] / n_simulations

        return {
            'home_win_prob': home_wins,
            'draw_prob': draws,
            'away_win_prob': away_wins,
            'common_scores': common_scores.to_dict(orient='records')
        }

    def get_team_strengths(self):
        if self.trace is None:
            return None
        summary = az.summary(self.trace, var_names=['attack_strength', 'defence_strength', 'home_advantage'])
        strengths = {}
        for i, team_name in self.id_team_map.items():
            strengths[team_name] = {
                'attack': summary.loc[f'attack_strength[{i}]']['mean'],
                'defence': summary.loc[f'defence_strength[{i}]']['mean'],
                'home_advantage': summary.loc[f'home_advantage[{i}]']['mean']
            }
        return strengths

# Global predictor instance
predictor = BayesianFootballPredictor()

# --- API Endpoints ---
@app.route('/train', methods=['POST'])
def train_model_endpoint():
    data = request.get_json()
    if not data or 'matches' not in data:
        return jsonify({"error": "Missing 'matches' data in request"}), 400

    try:
        # Example: data = {'matches': [{'home_team': 'A', 'away_team': 'B', 'home_score': 1, 'away_score': 0}, ...]}
        historical_df = pd.DataFrame(data['matches'])
        if not all(col in historical_df.columns for col in ['home_team', 'away_team', 'home_score', 'away_score']):
             return jsonify({"error": "Dataframe must contain 'home_team', 'away_team', 'home_score', 'away_score'"}), 400

        # Asynchronous training
        thread = threading.Thread(target=predictor.train, args=(historical_df,))
        thread.start()

        return jsonify({"message": "Model training started in background."}), 202
    except Exception as e:
        logger.error(f"Error during training endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    if not data or 'home_team' not in data or 'away_team' not in data:
        return jsonify({"error": "Request must include 'home_team' and 'away_team'"}), 400

    home_team = data['home_team']
    away_team = data['away_team']

    try:
        if predictor.trace is None:
             return jsonify({"error": "Model not trained yet. Please train the model first."}), 400
        predictions = predictor.predict_match_outcome_probs(home_team, away_team)
        return jsonify(predictions), 200
    except ValueError as ve: # Catch specific errors like team not found
        logger.warning(f"Prediction ValueError: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status_endpoint():
    return jsonify({
        "model_trained": predictor.trace is not None,
        "last_trained": predictor.last_trained_date.isoformat() if predictor.last_trained_date else None,
        "trained_teams_count": predictor.trained_teams,
        "trained_team_map_preview": dict(list(predictor.team_id_map.items())[:5]) # Preview first 5 teams
    }), 200

@app.route('/team_strengths', methods=['GET'])
def team_strengths_endpoint():
    strengths = predictor.get_team_strengths()
    if strengths:
        return jsonify(strengths), 200
    return jsonify({"error": "Model not trained or strengths not available."}), 400

def auto_retrain_model():
    """
    Periodically retrains the model if new data is available (placeholder logic).
    In a real system, this would fetch new match results.
    """
    while True:
        time.sleep(24 * 60 * 60) # Sleep for 24 hours
        logger.info("Checking for model retraining...")
        # Placeholder: Fetch new data. If significant new data, trigger training.
        # For demo, we'll just log. In a real scenario, you'd fetch from a DB or API.
        # Example: new_matches_df = fetch_new_match_data_from_db()
        # if not new_matches_df.empty:
        //    logger.info(f"Found {len(new_matches_df)} new matches. Retraining model.")
        //    predictor.train(new_matches_df) # This needs to be adapted for incremental training or full retrain
        logger.info("Auto-retrain check complete (no actual retraining in this placeholder).")


if __name__ == '__main__':
    # Example usage (typically you'd load data from a CSV or database)
    # This part would be removed if running as a managed service.
    # For local testing:
    if predictor.trace is None: # Only train if no model loaded/trained
        logger.info("No pre-trained model found. Training with sample data for local testing.")
        sample_data = {
            'matches': [
                {'home_team': 'TeamA', 'away_team': 'TeamB', 'home_score': 2, 'away_score': 1},
                {'home_team': 'TeamC', 'away_team': 'TeamD', 'home_score': 0, 'away_score': 0},
                {'home_team': 'TeamA', 'away_team': 'TeamC', 'home_score': 1, 'away_score': 1},
                {'home_team': 'TeamB', 'away_team': 'TeamD', 'home_score': 3, 'away_score': 2},
                # Add many more matches for a real model
            ] * 50 # Repeat for more data points
        }
        initial_df = pd.DataFrame(sample_data['matches'])

        # Start training in a separate thread to not block Flask app start for too long
        # In production, training might be a separate script or managed process.
        training_thread = threading.Thread(target=predictor.train, args=(initial_df,))
        training_thread.start()
        logger.info("Initial model training started in background for local testing.")

    # Start periodic retraining thread
    # retraining_thread = threading.Thread(target=auto_retrain_model, daemon=True)
    # retraining_thread.start()
    # logger.info("Automatic model retraining scheduler started.")

    app.run(host='0.0.0.0', port=5001, debug=False) # Set debug=False for production
```
