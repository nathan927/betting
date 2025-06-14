-- 001_initial_schema.sql
-- 初始資料庫架構設置

-- 啟用必要的擴展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- 用戶角色枚舉
CREATE TYPE user_role AS ENUM ('user', 'vip', 'admin', 'super_admin');

-- 賽事狀態枚舉
CREATE TYPE event_status AS ENUM ('scheduled', 'live', 'finished', 'cancelled', 'postponed');

-- 投注狀態枚舉
CREATE TYPE bet_status AS ENUM ('pending', 'won', 'lost', 'void', 'cashout');

-- 交易類型枚舉
CREATE TYPE transaction_type AS ENUM ('deposit', 'withdrawal', 'bet', 'payout', 'refund', 'bonus');

-- 用戶表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0.00 CHECK (balance >= 0),
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    verification_token VARCHAR(255),
    reset_password_token VARCHAR(255),
    reset_password_expires TIMESTAMPTZ,
    two_factor_secret VARCHAR(255),
    two_factor_enabled BOOLEAN DEFAULT false,
    last_login TIMESTAMPTZ,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 用戶索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_is_active ON users(is_active) WHERE is_active = true;

-- 用戶資料表
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    date_of_birth DATE,
    country VARCHAR(2),
    city VARCHAR(100),
    address TEXT,
    postal_code VARCHAR(20),
    preferred_language VARCHAR(5) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- KYC 驗證表
CREATE TABLE kyc_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL,
    document_number VARCHAR(100),
    document_country VARCHAR(2),
    verification_status VARCHAR(20) DEFAULT 'pending',
    verified_at TIMESTAMPTZ,
    verified_by UUID REFERENCES users(id),
    rejection_reason TEXT,
    document_front_url TEXT,
    document_back_url TEXT,
    selfie_url TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kyc_user_id ON kyc_verifications(user_id);
CREATE INDEX idx_kyc_status ON kyc_verifications(verification_status);

-- 賽事表
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sport_type VARCHAR(50) NOT NULL,
    competition VARCHAR(200),
    event_name VARCHAR(500) NOT NULL,
    home_team VARCHAR(200),
    away_team VARCHAR(200),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status event_status DEFAULT 'scheduled',
    venue VARCHAR(200),
    country VARCHAR(2),
    external_id VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 賽事索引
CREATE INDEX idx_events_sport_type ON events(sport_type);
CREATE INDEX idx_events_status ON events(status);
CREATE INDEX idx_events_start_time ON events(start_time);
CREATE INDEX idx_events_external_id ON events(external_id);
CREATE INDEX idx_events_metadata ON events USING GIN(metadata);

-- 市場類型表
CREATE TABLE market_types (
    id SERIAL PRIMARY KEY,
    sport_type VARCHAR(50) NOT NULL,
    market_code VARCHAR(50) NOT NULL,
    market_name VARCHAR(200) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    UNIQUE(sport_type, market_code)
);

-- 插入常見市場類型
INSERT INTO market_types (sport_type, market_code, market_name, description) VALUES
('football', '1X2', 'Match Result', 'Home Win, Draw, or Away Win'),
('football', 'BTTS', 'Both Teams to Score', 'Both teams score at least one goal'),
('football', 'O/U', 'Over/Under Goals', 'Total goals over or under specified line'),
('football', 'AH', 'Asian Handicap', 'Handicap betting with quarter/half goals'),
('horse_racing', 'WIN', 'Win', 'Horse to finish first'),
('horse_racing', 'PLACE', 'Place', 'Horse to finish in top positions'),
('horse_racing', 'EW', 'Each Way', 'Combination of Win and Place'),
('tennis', 'ML', 'Match Winner', 'Player to win the match'),
('basketball', 'SPREAD', 'Point Spread', 'Team to cover the spread');

-- 賠率歷史表（時間序列）
CREATE TABLE odds_history (
    odds_id UUID DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    bookmaker VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    selection VARCHAR(200) NOT NULL,
    odds_value DECIMAL(8,3) NOT NULL CHECK (odds_value > 1),
    lay_odds DECIMAL(8,3),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_available BOOLEAN DEFAULT true,
    volume DECIMAL(15,2),
    PRIMARY KEY (odds_id, timestamp)
);

-- 轉換為時間序列表
SELECT create_hypertable('odds_history', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- 賠率索引
CREATE INDEX idx_odds_event_time ON odds_history (event_id, timestamp DESC);
CREATE INDEX idx_odds_bookmaker_time ON odds_history (bookmaker, timestamp DESC);
CREATE INDEX idx_odds_market_selection ON odds_history (market_type, selection);

-- 套利機會表
CREATE TABLE arbitrage_opportunities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL REFERENCES events(event_id),
    opportunity_type VARCHAR(20) NOT NULL,
    profit_margin DECIMAL(5,4) NOT NULL CHECK (profit_margin > 0),
    bookmakers TEXT[] NOT NULL,
    selections TEXT[] NOT NULL,
    odds DECIMAL(8,3)[] NOT NULL,
    stakes DECIMAL(5,4)[] NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    detection_latency_ms INTEGER
);

CREATE INDEX idx_arbitrage_event ON arbitrage_opportunities(event_id);
CREATE INDEX idx_arbitrage_profit ON arbitrage_opportunities(profit_margin DESC);
CREATE INDEX idx_arbitrage_active ON arbitrage_opportunities(is_active) WHERE is_active = true;

-- 投注交易表（時間序列）
CREATE TABLE bet_transactions (
    transaction_id UUID DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    event_id UUID NOT NULL REFERENCES events(event_id),
    market_type VARCHAR(50) NOT NULL,
    selection VARCHAR(200) NOT NULL,
    bet_type VARCHAR(20) DEFAULT 'single',
    stake DECIMAL(10,2) NOT NULL CHECK (stake > 0),
    odds DECIMAL(8,3) NOT NULL CHECK (odds > 1),
    potential_payout DECIMAL(12,2) NOT NULL,
    actual_payout DECIMAL(12,2),
    bet_status bet_status DEFAULT 'pending',
    placed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMPTZ,
    cashout_value DECIMAL(10,2),
    cashout_at TIMESTAMPTZ,
    ip_address INET,
    device_info JSONB,
    PRIMARY KEY (transaction_id, placed_at)
);

-- 轉換為時間序列表
SELECT create_hypertable('bet_transactions', 'placed_at',
    chunk_time_interval => INTERVAL '1 day',
    partitioning_column => 'user_id',
    number_partitions => 32,
    if_not_exists => TRUE
);

-- 投注索引
CREATE INDEX idx_bets_user_time ON bet_transactions (user_id, placed_at DESC);
CREATE INDEX idx_bets_event ON bet_transactions (event_id);
CREATE INDEX idx_bets_status ON bet_transactions (bet_status);

-- 多重彩投注組合表
CREATE TABLE bet_combinations (
    combination_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    bet_type VARCHAR(20) NOT NULL,
    total_stake DECIMAL(10,2) NOT NULL,
    total_odds DECIMAL(12,3) NOT NULL,
    potential_payout DECIMAL(15,2) NOT NULL,
    combination_status bet_status DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bet_combination_legs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    combination_id UUID NOT NULL REFERENCES bet_combinations(combination_id) ON DELETE CASCADE,
    transaction_id UUID NOT NULL,
    leg_order INTEGER NOT NULL,
    UNIQUE(combination_id, leg_order)
);

-- 交易記錄表
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    type transaction_type NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    balance_before DECIMAL(15,2) NOT NULL,
    balance_after DECIMAL(15,2) NOT NULL,
    reference_id UUID,
    description TEXT,
    payment_method VARCHAR(50),
    payment_reference VARCHAR(200),
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_transactions_user ON transactions(user_id);
CREATE INDEX idx_transactions_type ON transactions(type);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_created ON transactions(created_at DESC);

-- 用戶限制表
CREATE TABLE user_limits (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    daily_deposit_limit DECIMAL(10,2),
    weekly_deposit_limit DECIMAL(10,2),
    monthly_deposit_limit DECIMAL(10,2),
    daily_loss_limit DECIMAL(10,2),
    weekly_loss_limit DECIMAL(10,2),
    monthly_loss_limit DECIMAL(10,2),
    session_time_limit INTEGER, -- 分鐘
    reality_check_interval INTEGER, -- 分鐘
    self_exclusion_until TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 審計日誌表
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);

-- 賽馬專用表
CREATE TABLE horses (
    horse_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    country VARCHAR(2),
    sire VARCHAR(200),
    dam VARCHAR(200),
    trainer_id UUID,
    owner VARCHAR(200),
    color VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE jockeys (
    jockey_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    nationality VARCHAR(2),
    weight DECIMAL(5,2),
    height DECIMAL(5,2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE trainers (
    trainer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    nationality VARCHAR(2),
    license_number VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE race_entries (
    entry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL REFERENCES events(event_id),
    horse_id UUID NOT NULL REFERENCES horses(horse_id),
    jockey_id UUID NOT NULL REFERENCES jockeys(jockey_id),
    trainer_id UUID NOT NULL REFERENCES trainers(trainer_id),
    barrier_draw INTEGER,
    weight_carried DECIMAL(5,2),
    race_number INTEGER,
    is_scratched BOOLEAN DEFAULT false,
    scratched_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_race_entries_event ON race_entries(event_id);
CREATE INDEX idx_race_entries_horse ON race_entries(horse_id);

-- 賽馬表現歷史
CREATE TABLE horse_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    horse_id UUID NOT NULL REFERENCES horses(horse_id),
    event_id UUID NOT NULL REFERENCES events(event_id),
    finish_position INTEGER,
    finish_time DECIMAL(8,2),
    lengths_behind DECIMAL(5,2),
    speed_figure DECIMAL(5,2),
    track_condition VARCHAR(20),
    race_distance INTEGER,
    race_class VARCHAR(20),
    prize_money DECIMAL(10,2),
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_horse ON horse_performance(horse_id);
CREATE INDEX idx_performance_event ON horse_performance(event_id);

-- 機器學習模型元數據
CREATE TABLE ml_models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    accuracy DECIMAL(5,4),
    parameters JSONB,
    training_data_size INTEGER,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT false,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version)
);

CREATE TABLE ml_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(model_id),
    event_id UUID NOT NULL REFERENCES events(event_id),
    prediction_type VARCHAR(50),
    predictions JSONB NOT NULL,
    confidence DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ml_predictions_model ON ml_predictions(model_id);
CREATE INDEX idx_ml_predictions_event ON ml_predictions(event_id);

-- 促銷活動表
CREATE TABLE promotions (
    promotion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    min_deposit DECIMAL(10,2),
    min_odds DECIMAL(5,2),
    wagering_requirement INTEGER,
    max_uses INTEGER,
    uses_count INTEGER DEFAULT 0,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_promotions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    promotion_id UUID NOT NULL REFERENCES promotions(promotion_id),
    used_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    wagering_completed DECIMAL(10,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    UNIQUE(user_id, promotion_id)
);

-- 通知表
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_read BOOLEAN DEFAULT false,
    read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_unread ON notifications(user_id) WHERE is_read = false;

-- 壓縮策略（針對時間序列表）
ALTER TABLE odds_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'event_id, bookmaker',
    timescaledb.compress_orderby = 'timestamp DESC'
);

ALTER TABLE bet_transactions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id',
    timescaledb.compress_orderby = 'placed_at DESC'
);

-- 添加壓縮策略
SELECT add_compression_policy('odds_history', INTERVAL '1 day');
SELECT add_compression_policy('bet_transactions', INTERVAL '7 days');

-- 添加數據保留策略
SELECT add_retention_policy('odds_history', INTERVAL '90 days');
SELECT add_retention_policy('bet_transactions', INTERVAL '365 days');

-- 創建連續聚合視圖（用於快速查詢）
CREATE MATERIALIZED VIEW hourly_betting_stats
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', placed_at) AS hour,
    COUNT(*) as total_bets,
    SUM(stake) as total_stake,
    AVG(odds) as avg_odds,
    COUNT(DISTINCT user_id) as unique_users
FROM bet_transactions
GROUP BY hour
WITH NO DATA;

-- 設置連續聚合的刷新策略
SELECT add_continuous_aggregate_policy('hourly_betting_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- 觸發器：更新 updated_at 時間戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 函數：計算用戶統計
CREATE OR REPLACE FUNCTION get_user_stats(p_user_id UUID)
RETURNS TABLE(
    total_bets BIGINT,
    total_stake DECIMAL,
    total_payout DECIMAL,
    win_rate DECIMAL,
    avg_odds DECIMAL,
    favorite_sport VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_bets,
        COALESCE(SUM(bt.stake), 0) as total_stake,
        COALESCE(SUM(bt.actual_payout), 0) as total_payout,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                COUNT(CASE WHEN bt.bet_status = 'won' THEN 1 END)::DECIMAL / COUNT(*)::DECIMAL
            ELSE 0 
        END as win_rate,
        AVG(bt.odds) as avg_odds,
        MODE() WITHIN GROUP (ORDER BY e.sport_type) as favorite_sport
    FROM bet_transactions bt
    JOIN events e ON bt.event_id = e.event_id
    WHERE bt.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- 函數：檢查用戶投注限制
CREATE OR REPLACE FUNCTION check_user_betting_limits(
    p_user_id UUID,
    p_amount DECIMAL
) RETURNS BOOLEAN AS $$
DECLARE
    v_daily_total DECIMAL;
    v_weekly_total DECIMAL;
    v_monthly_total DECIMAL;
    v_limits RECORD;
BEGIN
    -- 獲取用戶限制
    SELECT * INTO v_limits FROM user_limits WHERE user_id = p_user_id;
    
    IF NOT FOUND THEN
        RETURN TRUE; -- 沒有設置限制
    END IF;
    
    -- 檢查自我排除
    IF v_limits.self_exclusion_until IS NOT NULL AND v_limits.self_exclusion_until > NOW() THEN
        RETURN FALSE;
    END IF;
    
    -- 計算當前週期內的投注總額
    SELECT COALESCE(SUM(stake), 0) INTO v_daily_total
    FROM bet_transactions
    WHERE user_id = p_user_id 
    AND placed_at >= CURRENT_DATE;
    
    SELECT COALESCE(SUM(stake), 0) INTO v_weekly_total
    FROM bet_transactions
    WHERE user_id = p_user_id 
    AND placed_at >= DATE_TRUNC('week', CURRENT_DATE);
    
    SELECT COALESCE(SUM(stake), 0) INTO v_monthly_total
    FROM bet_transactions
    WHERE user_id = p_user_id 
    AND placed_at >= DATE_TRUNC('month', CURRENT_DATE);
    
    -- 檢查限制
    IF v_limits.daily_deposit_limit IS NOT NULL AND v_daily_total + p_amount > v_limits.daily_deposit_limit THEN
        RETURN FALSE;
    END IF;
    
    IF v_limits.weekly_deposit_limit IS NOT NULL AND v_weekly_total + p_amount > v_limits.weekly_deposit_limit THEN
        RETURN FALSE;
    END IF;
    
    IF v_limits.monthly_deposit_limit IS NOT NULL AND v_monthly_total + p_amount > v_limits.monthly_deposit_limit THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- 創建角色和權限
CREATE ROLE betting_app LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE betting_db TO betting_app;
GRANT USAGE ON SCHEMA public TO betting_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO betting_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO betting_app;

-- 只讀角色（用於報表）
CREATE ROLE betting_readonly LOGIN PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE betting_db TO betting_readonly;
GRANT USAGE ON SCHEMA public TO betting_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO betting_readonly;

-- 管理員角色
CREATE ROLE betting_admin LOGIN PASSWORD 'admin_password';
GRANT ALL PRIVILEGES ON DATABASE betting_db TO betting_admin;