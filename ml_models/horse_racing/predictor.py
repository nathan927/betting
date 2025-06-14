# horse_racing_prediction.py - 生產級賽馬預測系統
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
import joblib
import json
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HorseRacingFeatureEngineering:
    """賽馬特徵工程類"""
    
    def __init__(self):
        self.feature_columns = [
            'horse_age', 'horse_weight', 'jockey_win_rate', 'trainer_win_rate',
            'barrier_draw', 'class_rating', 'speed_figure_avg', 'days_since_last_race',
            'track_condition_encoded', 'distance_category', 'race_class', 'weight_carried',
            'recent_form_score', 'course_specialist_score', 'distance_specialist_score',
            'jockey_trainer_combo_rate', 'going_preference_score', 'rest_days_optimal',
            'weight_change', 'class_change', 'handicap_rating', 'prize_money_avg',
            'finish_position_avg', 'beaten_lengths_avg', 'sectional_time_rating',
            'early_speed_rating', 'late_speed_rating', 'pace_rating'
        ]
        
        self.scalers = {}
        self.encoders = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """執行完整的特徵工程"""
        logger.info(f"Starting feature engineering for {len(df)} records")
        
        # 基礎特徵
        df = self._create_basic_features(df)
        
        # 表現特徵
        df = self._create_performance_features(df)
        
        # 組合特徵
        df = self._create_combination_features(df)
        
        # 趨勢特徵
        df = self._create_trend_features(df)
        
        # 速度特徵
        df = self._create_speed_features(df)
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_columns)}")
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建基礎特徵"""
        # 馬匹年齡（考慮最佳年齡區間）
        df['age_optimal'] = df['horse_age'].apply(lambda x: 1 if 4 <= x <= 7 else 0)
        
        # 負重相對值
        df['weight_relative'] = df['weight_carried'] / df.groupby('race_id')['weight_carried'].transform('mean')
        
        # 休息天數分類
        df['rest_category'] = pd.cut(df['days_since_last_race'], 
                                     bins=[0, 14, 30, 60, 365], 
                                     labels=['short', 'normal', 'long', 'very_long'])
        
        # 檔位優勢
        df['draw_advantage'] = df.apply(self._calculate_draw_advantage, axis=1)
        
        return df
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建表現相關特徵"""
        # 近期表現評分（最近5場）
        df['recent_form_score'] = df.groupby('horse_id').apply(
            lambda x: self._calculate_form_score(x.tail(5))
        ).reset_index(level=0, drop=True)
        
        # 場地專家評分
        df['course_specialist_score'] = df.groupby(['horse_id', 'course_id']).apply(
            lambda x: self._calculate_specialist_score(x)
        ).reset_index(level=[0, 1], drop=True)
        
        # 距離專家評分
        df['distance_specialist_score'] = df.groupby(['horse_id', 'distance_category']).apply(
            lambda x: self._calculate_distance_score(x)
        ).reset_index(level=[0, 1], drop=True)
        
        return df
    
    def _create_combination_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建組合特徵"""
        # 騎師-練馬師組合勝率
        jt_stats = df.groupby(['jockey_id', 'trainer_id']).agg({
            'win': 'mean',
            'place': 'mean',
            'race_id': 'count'
        }).reset_index()
        jt_stats.columns = ['jockey_id', 'trainer_id', 'jt_win_rate', 'jt_place_rate', 'jt_races']
        
        df = df.merge(jt_stats, on=['jockey_id', 'trainer_id'], how='left')
        df['jockey_trainer_combo_rate'] = df['jt_win_rate'].fillna(df['jockey_win_rate'])
        
        # 馬匹-騎師配合度
        horse_jockey_stats = df.groupby(['horse_id', 'jockey_id']).agg({
            'win': 'mean',
            'finish_position': 'mean'
        }).reset_index()
        horse_jockey_stats.columns = ['horse_id', 'jockey_id', 'hj_win_rate', 'hj_avg_position']
        
        df = df.merge(horse_jockey_stats, on=['horse_id', 'jockey_id'], how='left')
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建趨勢特徵"""
        # 速度趨勢
        df['speed_trend'] = df.groupby('horse_id')['speed_figure'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 排名趨勢
        df['position_trend'] = df.groupby('horse_id')['finish_position'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean()
        )
        
        # 級別變化
        df['class_change'] = df.groupby('horse_id')['race_class'].diff()
        
        return df
    
    def _create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建速度相關特徵"""
        # 分段時間評級
        if 'sectional_times' in df.columns:
            df['early_speed_rating'] = df['sectional_times'].apply(
                lambda x: self._calculate_early_speed(x) if pd.notna(x) else 0
            )
            df['late_speed_rating'] = df['sectional_times'].apply(
                lambda x: self._calculate_late_speed(x) if pd.notna(x) else 0
            )
        
        # 步速評級
        df['pace_rating'] = df.apply(self._calculate_pace_rating, axis=1)
        
        return df
    
    def _calculate_draw_advantage(self, row):
        """計算檔位優勢"""
        distance = row['distance']
        barrier = row['barrier_draw']
        
        if distance <= 1200:  # 短途
            return 1 - abs(barrier - 3) / 10
        elif distance <= 1600:  # 中距離
            return 1 - abs(barrier - 6) / 10
        else:  # 長途
            return 1 - abs(barrier - 8) / 10
    
    def _calculate_form_score(self, recent_races):
        """計算近期表現評分"""
        if len(recent_races) == 0:
            return 0
        
        weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05])[:len(recent_races)]
        positions = recent_races['finish_position'].values
        
        # 轉換排名為分數（第1名=1.0, 第2名=0.8, 等等）
        scores = np.maximum(0, 1 - (positions - 1) * 0.2)
        
        return np.sum(scores * weights) / np.sum(weights)
    
    def _calculate_specialist_score(self, group_data):
        """計算場地專家評分"""
        if len(group_data) < 3:
            return 0.5
        
        win_rate = group_data['win'].mean()
        place_rate = group_data['place'].mean()
        avg_position = group_data['finish_position'].mean()
        
        score = (win_rate * 0.5 + place_rate * 0.3 + (1 - avg_position/10) * 0.2)
        return np.clip(score, 0, 1)
    
    def _calculate_distance_score(self, group_data):
        """計算距離專家評分"""
        if len(group_data) < 2:
            return 0.5
        
        win_rate = group_data['win'].mean()
        avg_beaten_lengths = group_data['beaten_lengths'].mean() if 'beaten_lengths' in group_data else 5
        
        score = win_rate * 0.7 + (1 - min(avg_beaten_lengths, 10) / 10) * 0.3
        return np.clip(score, 0, 1)
    
    def _calculate_early_speed(self, sectional_times):
        """計算前段速度評級"""
        if not sectional_times or len(sectional_times) < 2:
            return 0.5
        
        early_sections = sectional_times[:len(sectional_times)//2]
        return 1 / (1 + np.mean(early_sections))
    
    def _calculate_late_speed(self, sectional_times):
        """計算後段速度評級"""
        if not sectional_times or len(sectional_times) < 2:
            return 0.5
        
        late_sections = sectional_times[len(sectional_times)//2:]
        return 1 / (1 + np.mean(late_sections))
    
    def _calculate_pace_rating(self, row):
        """計算步速評級"""
        if 'stride_length' in row and 'stride_frequency' in row:
            return row['stride_length'] * row['stride_frequency'] / 100
        return 0.5

class HorseRacingNeuralNetwork(nn.Module):
    """深度神經網絡模型"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(HorseRacingNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 輸出層
        layers.extend([
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class HorseRacingPredictor:
    """賽馬預測主類"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_engineer = HorseRacingFeatureEngineering()
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"Predictor initialized. Using device: {self.device}")
    
    def train(self, train_data: pd.DataFrame, epochs: int = 200, batch_size: int = 64):
        """訓練模型"""
        logger.info("Starting model training")
        
        # 特徵工程
        train_data = self.feature_engineer.engineer_features(train_data)
        
        # 準備訓練數據
        X = train_data[self.feature_engineer.feature_columns].values
        y = train_data['win'].values
        
        # 分割數據
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # 標準化
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # 轉換為張量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 創建數據加載器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        self.model = HorseRacingNeuralNetwork(
            input_dim=X_train.shape[1]
        ).to(self.device)
        
        # 設置優化器和損失函數
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # 計算正樣本權重（處理類別不平衡）
        pos_weight = torch.tensor([len(y_train) / sum(y_train) - 1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 訓練循環
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor).squeeze()
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
                
                # 計算AUC
                val_auc = roc_auc_score(
                    y_val_tensor.cpu().numpy(),
                    val_outputs.cpu().numpy()
                )
            
            # 學習率調整
            scheduler.step(val_accuracy)
            
            # 早停檢查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                self._save_checkpoint(epoch, val_accuracy, val_auc)
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss/len(train_loader):.4f} - "
                    f"Val Accuracy: {val_accuracy:.3f} - "
                    f"Val AUC: {val_auc:.3f}"
                )
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.3f}")
        return best_val_accuracy
    
    def predict(self, race_data: pd.DataFrame) -> Dict[str, float]:
        """預測單場比賽"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # 特徵工程
        race_data = self.feature_engineer.engineer_features(race_data)
        
        # 準備數據
        X = race_data[self.feature_engineer.feature_columns].values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 預測
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_tensor).squeeze().cpu().numpy()
        
        # 整理結果
        results = {}
        for idx, row in race_data.iterrows():
            horse_number = row['horse_number']
            horse_name = row['horse_name']
            win_probability = float(probabilities[idx])
            
            # 計算建議投注金額（Kelly準則）
            suggested_stake = self._calculate_kelly_stake(
                win_probability, 
                row.get('odds', 3.0)
            )
            
            results[horse_number] = {
                'horse_name': horse_name,
                'win_probability': win_probability,
                'rank': 0,  # 將在後續填充
                'suggested_stake': suggested_stake,
                'expected_value': win_probability * row.get('odds', 3.0) - 1
            }
        
        # 排名
        sorted_horses = sorted(
            results.items(), 
            key=lambda x: x[1]['win_probability'], 
            reverse=True
        )
        
        for rank, (horse_num, _) in enumerate(sorted_horses, 1):
            results[horse_num]['rank'] = rank
        
        return results
    
    def _calculate_kelly_stake(self, win_prob: float, odds: float, kelly_fraction: float = 0.25):
        """使用Kelly準則計算建議投注金額"""
        if win_prob <= 0 or odds <= 1:
            return 0
        
        # Kelly公式: f = (p * b - q) / b
        # 其中 p = 獲勝概率, q = 失敗概率, b = 淨賠率
        q = 1 - win_prob
        b = odds - 1
        
        kelly_stake = (win_prob * b - q) / b
        
        # 使用部分Kelly（更保守）
        kelly_stake = kelly_stake * kelly_fraction
        
        # 限制在合理範圍內
        return max(0, min(kelly_stake, 0.1))  # 最多10%資金
    
    def _save_checkpoint(self, epoch: int, accuracy: float, auc: float):
        """保存模型檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'auc': auc,
            'scaler': self.scaler,
            'feature_columns': self.feature_engineer.feature_columns
        }
        
        filename = f'horse_racing_model_acc{accuracy:.3f}_auc{auc:.3f}.pth'
        torch.save(checkpoint, filename)
        logger.info(f"Model checkpoint saved: {filename}")
    
    def load_model(self, model_path: str):
        """載入模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 重建模型
        input_dim = len(checkpoint['feature_columns'])
        self.model = HorseRacingNeuralNetwork(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 載入其他組件
        self.scaler = checkpoint['scaler']
        self.feature_engineer.feature_columns = checkpoint['feature_columns']
        
        logger.info(f"Model loaded. Accuracy: {checkpoint['accuracy']:.3f}")
    
    def evaluate_performance(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """評估模型性能"""
        # 特徵工程
        test_data = self.feature_engineer.engineer_features(test_data)
        
        # 按比賽分組評估
        performance_metrics = {
            'overall_accuracy': 0,
            'top3_accuracy': 0,
            'average_roi': 0,
            'profitable_races': 0,
            'total_races': 0
        }
        
        race_ids = test_data['race_id'].unique()
        
        for race_id in race_ids:
            race_data = test_data[test_data['race_id'] == race_id]
            predictions = self.predict(race_data)
            
            # 評估這場比賽的預測
            race_metrics = self._evaluate_single_race(race_data, predictions)
            
            # 更新總體指標
            performance_metrics['total_races'] += 1
            if race_metrics['is_winner_predicted']:
                performance_metrics['overall_accuracy'] += 1
            if race_metrics['is_top3_predicted']:
                performance_metrics['top3_accuracy'] += 1
            if race_metrics['roi'] > 0:
                performance_metrics['profitable_races'] += 1
            performance_metrics['average_roi'] += race_metrics['roi']
        
        # 計算平均值
        if performance_metrics['total_races'] > 0:
            performance_metrics['overall_accuracy'] /= performance_metrics['total_races']
            performance_metrics['top3_accuracy'] /= performance_metrics['total_races']
            performance_metrics['average_roi'] /= performance_metrics['total_races']
            performance_metrics['profitable_races'] /= performance_metrics['total_races']
        
        logger.info(f"Model evaluation completed: {performance_metrics}")
        return performance_metrics
    
    def _evaluate_single_race(self, race_data: pd.DataFrame, predictions: Dict) -> Dict:
        """評估單場比賽的預測結果"""
        # 找出實際獲勝者
        actual_winner = race_data[race_data['win'] == 1]['horse_number'].values[0]
        
        # 找出預測的前三名
        top_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1]['win_probability'],
            reverse=True
        )[:3]
        
        predicted_winner = top_predictions[0][0]
        predicted_top3 = [h[0] for h in top_predictions]
        
        # 計算投資回報率（假設投注預測第一名）
        bet_amount = 100  # 假設固定投注額
        if predicted_winner == actual_winner:
            winner_odds = race_data[race_data['horse_number'] == actual_winner]['odds'].values[0]
            roi = (winner_odds - 1) * bet_amount
        else:
            roi = -bet_amount
        
        return {
            'is_winner_predicted': predicted_winner == actual_winner,
            'is_top3_predicted': actual_winner in predicted_top3,
            'roi': roi / bet_amount,
            'predicted_winner_prob': predictions[predicted_winner]['win_probability']
        }

class HKJCDataCollector:
    """香港賽馬會數據收集器"""
    
    def __init__(self):
        self.base_url = "https://bet.hkjc.com/racing/getJSON.aspx"
        self.session = None
    
    async def collect_race_data(self, race_date: str, venue: str) -> pd.DataFrame:
        """收集指定日期和場地的賽事數據"""
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # 獲取賽事列表
            races = await self._get_race_list(race_date, venue)
            
            all_race_data = []
            
            for race in races:
                race_id = race['race_id']
                race_data = await self._get_race_details(race_id)
                
                if race_data:
                    all_race_data.extend(race_data)
            
            return pd.DataFrame(all_race_data)
    
    async def _get_race_list(self, race_date: str, venue: str) -> List[Dict]:
        """獲取賽事列表"""
        # 這裡應該實現實際的API調用
        # 為了示例，返回模擬數據
        return [
            {'race_id': f"{race_date}_{venue}_R{i}", 'race_number': i}
            for i in range(1, 11)
        ]
    
    async def _get_race_details(self, race_id: str) -> List[Dict]:
        """獲取單場賽事詳情"""
        # 這裡應該實現實際的API調用
        # 為了示例，返回模擬數據
        horses = []
        for i in range(1, 15):
            horses.append({
                'race_id': race_id,
                'horse_number': i,
                'horse_name': f"Horse_{i}",
                'horse_age': np.random.randint(3, 8),
                'weight_carried': np.random.uniform(115, 135),
                'barrier_draw': i,
                'jockey_win_rate': np.random.uniform(0.05, 0.25),
                'trainer_win_rate': np.random.uniform(0.08, 0.22),
                'odds': np.random.uniform(2.5, 50.0),
                'win': 1 if i == np.random.randint(1, 15) else 0
            })
        
        return horses

# 主程序
async def main():
    """主程序入口"""
    # 初始化預測器
    predictor = HorseRacingPredictor()
    
    # 收集數據
    collector = HKJCDataCollector()
    training_data = await collector.collect_race_data("2024-01-15", "HV")
    
    # 訓練模型
    if len(training_data) > 0:
        accuracy = predictor.train(training_data, epochs=100)
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        
        # 預測示例
        test_race_data = await collector.collect_race_data("2024-01-16", "ST")
        if len(test_race_data) > 0:
            race_1_data = test_race_data[test_race_data['race_id'] == test_race_data['race_id'].iloc[0]]
            predictions = predictor.predict(race_1_data)
            
            print("\n=== 賽馬預測結果 ===")
            for horse_num, pred in sorted(predictions.items(), key=lambda x: x[1]['rank']):
                print(f"排名 {pred['rank']}: {pred['horse_name']} "
                      f"(勝率: {pred['win_probability']:.3f}, "
                      f"建議投注: {pred['suggested_stake']*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())