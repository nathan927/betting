// arbitrage-detection-engine.js - 生產級套利檢測系統
const EventEmitter = require('events');
const { Kafka } = require('kafkajs');
const Redis = require('redis');
const { Pool } = require('pg');
const winston = require('winston');
const promClient = require('prom-client');

// 配置日誌
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    defaultMeta: { service: 'arbitrage-engine' },
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/arbitrage.log' })
    ]
});

// 套利檢測引擎主類
class ArbitrageDetectionEngine extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            minProfitThreshold: config.minProfitThreshold || 0.02,
            maxStakeLimit: config.maxStakeLimit || 10000,
            oddsTolerance: config.oddsTolerance || 0.001,
            detectionWindowMs: config.detectionWindowMs || 500,
            bookmakerDelayMs: config.bookmakerDelayMs || 100,
            ...config
        };
        
        this.kafka = null;
        this.redis = null;
        this.db = null;
        this.oddsCache = new Map();
        this.bookmakerLatency = new Map();
        this.metrics = this.setupMetrics();
        this.isRunning = false;
    }

    setupMetrics() {
        return {
            arbitrageDetected: new promClient.Counter({
                name: 'arbitrage_opportunities_detected_total',
                help: 'Total number of arbitrage opportunities detected',
                labelNames: ['type', 'profit_range']
            }),
            detectionLatency: new promClient.Histogram({
                name: 'arbitrage_detection_latency_ms',
                help: 'Arbitrage detection latency in milliseconds',
                buckets: [10, 50, 100, 200, 500, 1000, 2000]
            }),
            profitMargin: new promClient.Histogram({
                name: 'arbitrage_profit_margin',
                help: 'Distribution of arbitrage profit margins',
                buckets: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
            }),
            oddsProcessed: new promClient.Counter({
                name: 'odds_updates_processed_total',
                help: 'Total number of odds updates processed',
                labelNames: ['bookmaker']
            })
        };
    }

    async initialize() {
        try {
            // 初始化 Kafka
            this.kafka = new Kafka({
                clientId: 'arbitrage-detector',
                brokers: process.env.KAFKA_BROKERS?.split(',') || ['localhost:9092'],
                retry: {
                    initialRetryTime: 100,
                    retries: 8
                }
            });

            // 初始化 Redis
            this.redis = Redis.createClient({
                host: process.env.REDIS_HOST || 'localhost',
                port: process.env.REDIS_PORT || 6379,
                password: process.env.REDIS_PASSWORD
            });

            // 初始化資料庫
            this.db = new Pool({
                host: process.env.DB_HOST || 'localhost',
                port: process.env.DB_PORT || 5432,
                database: process.env.DB_NAME || 'betting_db',
                user: process.env.DB_USER || 'betting_user',
                password: process.env.DB_PASSWORD
            });

            await this.setupKafkaConsumer();
            await this.loadBookmakerConfig();
            
            logger.info('Arbitrage detection engine initialized');
        } catch (error) {
            logger.error('Failed to initialize arbitrage engine:', error);
            throw error;
        }
    }

    async setupKafkaConsumer() {
        this.consumer = this.kafka.consumer({ 
            groupId: 'arbitrage-detection-group',
            sessionTimeout: 30000,
            heartbeatInterval: 3000
        });

        await this.consumer.connect();
        await this.consumer.subscribe({ 
            topics: ['odds-updates'], 
            fromBeginning: false 
        });

        this.producer = this.kafka.producer({
            maxInFlightRequests: 5,
            idempotent: true,
            transactionalId: 'arbitrage-producer'
        });

        await this.producer.connect();
    }

    async loadBookmakerConfig() {
        const result = await this.db.query(`
            SELECT DISTINCT bookmaker, 
                   AVG(processing_time_ms) as avg_latency
            FROM bookmaker_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY bookmaker
        `);

        result.rows.forEach(row => {
            this.bookmakerLatency.set(row.bookmaker, row.avg_latency || 100);
        });
    }

    async start() {
        if (this.isRunning) {
            logger.warn('Arbitrage engine already running');
            return;
        }

        this.isRunning = true;
        logger.info('Starting arbitrage detection engine');

        // 啟動 Kafka 消費者
        await this.consumer.run({
            eachMessage: async ({ topic, partition, message }) => {
                await this.processOddsUpdate(message);
            }
        });

        // 定期清理過期數據
        this.cleanupInterval = setInterval(() => {
            this.cleanupExpiredOdds();
        }, 30000);

        // 定期保存統計數據
        this.statsInterval = setInterval(() => {
            this.saveStatistics();
        }, 60000);
    }

    async stop() {
        this.isRunning = false;
        
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
        }

        await this.consumer.disconnect();
        await this.producer.disconnect();
        
        logger.info('Arbitrage detection engine stopped');
    }

    async processOddsUpdate(message) {
        const startTime = process.hrtime.bigint();
        
        try {
            const oddsData = JSON.parse(message.value.toString());
            this.metrics.oddsProcessed.inc({ bookmaker: oddsData.bookmaker });

            // 更新賠率緩存
            this.updateOddsCache(oddsData);

            // 檢測套利機會
            const opportunities = await this.detectArbitrageOpportunities(oddsData.eventId);

            // 記錄檢測延遲
            const latency = Number(process.hrtime.bigint() - startTime) / 1_000_000;
            this.metrics.detectionLatency.observe(latency);

            // 發布套利機會
            for (const opportunity of opportunities) {
                if (opportunity.profitMargin >= this.config.minProfitThreshold) {
                    await this.publishArbitrageOpportunity(opportunity);
                }
            }
        } catch (error) {
            logger.error('Error processing odds update:', error);
        }
    }

    updateOddsCache(oddsData) {
        const { eventId, bookmaker, marketType, selection, odds, timestamp } = oddsData;
        
        const cacheKey = `${eventId}:${marketType}`;
        if (!this.oddsCache.has(cacheKey)) {
            this.oddsCache.set(cacheKey, new Map());
        }

        const marketCache = this.oddsCache.get(cacheKey);
        const selectionKey = `${bookmaker}:${selection}`;
        
        marketCache.set(selectionKey, {
            bookmaker,
            selection,
            odds,
            timestamp,
            expiresAt: Date.now() + this.config.detectionWindowMs
        });
    }

    async detectArbitrageOpportunities(eventId) {
        const opportunities = [];
        
        // 獲取該賽事的所有市場
        const markets = Array.from(this.oddsCache.keys())
            .filter(key => key.startsWith(`${eventId}:`));

        for (const marketKey of markets) {
            const marketOdds = this.oddsCache.get(marketKey);
            const marketType = marketKey.split(':')[1];
            
            // 根據市場類型檢測套利
            if (marketType === '1X2') {
                const threeWayOpp = this.detectThreeWayArbitrage(marketOdds);
                if (threeWayOpp) {
                    opportunities.push({
                        ...threeWayOpp,
                        eventId,
                        marketType
                    });
                }
            } else if (marketType === 'ML' || marketType === 'WIN') {
                const twoWayOpp = this.detectTwoWayArbitrage(marketOdds);
                if (twoWayOpp) {
                    opportunities.push({
                        ...twoWayOpp,
                        eventId,
                        marketType
                    });
                }
            } else if (marketType === 'O/U') {
                const overUnderOpp = this.detectOverUnderArbitrage(marketOdds);
                if (overUnderOpp) {
                    opportunities.push({
                        ...overUnderOpp,
                        eventId,
                        marketType
                    });
                }
            }
        }

        return opportunities;
    }

    detectTwoWayArbitrage(marketOdds) {
        const oddsArray = Array.from(marketOdds.values())
            .filter(o => Date.now() < o.expiresAt);

        if (oddsArray.length < 2) return null;

        // 按選項分組
        const outcome1Odds = oddsArray.filter(o => o.selection === 'HOME' || o.selection === '1');
        const outcome2Odds = oddsArray.filter(o => o.selection === 'AWAY' || o.selection === '2');

        if (outcome1Odds.length === 0 || outcome2Odds.length === 0) return null;

        // 找出最佳賠率
        const bestOdds1 = Math.max(...outcome1Odds.map(o => o.odds));
        const bestOdds2 = Math.max(...outcome2Odds.map(o => o.odds));

        const bestBookmaker1 = outcome1Odds.find(o => o.odds === bestOdds1).bookmaker;
        const bestBookmaker2 = outcome2Odds.find(o => o.odds === bestOdds2).bookmaker;

        // 計算套利
        const impliedProb1 = 1 / bestOdds1;
        const impliedProb2 = 1 / bestOdds2;
        const totalImpliedProb = impliedProb1 + impliedProb2;

        if (totalImpliedProb < 1 - this.config.oddsTolerance) {
            const profitMargin = (1 - totalImpliedProb) / totalImpliedProb;
            
            // 計算最優投注比例
            const stake1 = impliedProb1 / totalImpliedProb;
            const stake2 = impliedProb2 / totalImpliedProb;

            // 記錄指標
            this.metrics.arbitrageDetected.inc({ 
                type: 'two-way', 
                profit_range: this.getProfitRange(profitMargin) 
            });
            this.metrics.profitMargin.observe(profitMargin);

            return {
                type: 'two-way',
                profitMargin,
                bookmakers: [bestBookmaker1, bestBookmaker2],
                selections: ['HOME/1', 'AWAY/2'],
                odds: [bestOdds1, bestOdds2],
                stakes: [stake1, stake2],
                impliedProbability: totalImpliedProb,
                expectedReturn: 1 / totalImpliedProb,
                detectedAt: new Date().toISOString()
            };
        }

        return null;
    }

    detectThreeWayArbitrage(marketOdds) {
        const oddsArray = Array.from(marketOdds.values())
            .filter(o => Date.now() < o.expiresAt);

        if (oddsArray.length < 3) return null;

        // 按選項分組
        const homeOdds = oddsArray.filter(o => o.selection === 'HOME' || o.selection === '1');
        const drawOdds = oddsArray.filter(o => o.selection === 'DRAW' || o.selection === 'X');
        const awayOdds = oddsArray.filter(o => o.selection === 'AWAY' || o.selection === '2');

        if (homeOdds.length === 0 || drawOdds.length === 0 || awayOdds.length === 0) {
            return null;
        }

        // 找出最佳賠率
        const bestHomeOdds = Math.max(...homeOdds.map(o => o.odds));
        const bestDrawOdds = Math.max(...drawOdds.map(o => o.odds));
        const bestAwayOdds = Math.max(...awayOdds.map(o => o.odds));

        const bestHomeBookmaker = homeOdds.find(o => o.odds === bestHomeOdds).bookmaker;
        const bestDrawBookmaker = drawOdds.find(o => o.odds === bestDrawOdds).bookmaker;
        const bestAwayBookmaker = awayOdds.find(o => o.odds === bestAwayOdds).bookmaker;

        // 計算套利
        const impliedProbHome = 1 / bestHomeOdds;
        const impliedProbDraw = 1 / bestDrawOdds;
        const impliedProbAway = 1 / bestAwayOdds;
        const totalImpliedProb = impliedProbHome + impliedProbDraw + impliedProbAway;

        if (totalImpliedProb < 1 - this.config.oddsTolerance) {
            const profitMargin = (1 - totalImpliedProb) / totalImpliedProb;
            
            // 計算最優投注比例
            const stakeHome = impliedProbHome / totalImpliedProb;
            const stakeDraw = impliedProbDraw / totalImpliedProb;
            const stakeAway = impliedProbAway / totalImpliedProb;

            // 記錄指標
            this.metrics.arbitrageDetected.inc({ 
                type: 'three-way', 
                profit_range: this.getProfitRange(profitMargin) 
            });
            this.metrics.profitMargin.observe(profitMargin);

            return {
                type: 'three-way',
                profitMargin,
                bookmakers: [bestHomeBookmaker, bestDrawBookmaker, bestAwayBookmaker],
                selections: ['HOME', 'DRAW', 'AWAY'],
                odds: [bestHomeOdds, bestDrawOdds, bestAwayOdds],
                stakes: [stakeHome, stakeDraw, stakeAway],
                impliedProbability: totalImpliedProb,
                expectedReturn: 1 / totalImpliedProb,
                detectedAt: new Date().toISOString()
            };
        }

        return null;
    }

    detectOverUnderArbitrage(marketOdds) {
        const oddsArray = Array.from(marketOdds.values())
            .filter(o => Date.now() < o.expiresAt);

        if (oddsArray.length < 2) return null;

        // 按大小盤分組
        const overOdds = oddsArray.filter(o => o.selection.startsWith('OVER'));
        const underOdds = oddsArray.filter(o => o.selection.startsWith('UNDER'));

        // 找出相同盤口的最佳賠率
        const lineGroups = new Map();

        overOdds.forEach(o => {
            const line = o.selection.replace('OVER_', '');
            if (!lineGroups.has(line)) {
                lineGroups.set(line, { over: [], under: [] });
            }
            lineGroups.get(line).over.push(o);
        });

        underOdds.forEach(o => {
            const line = o.selection.replace('UNDER_', '');
            if (lineGroups.has(line)) {
                lineGroups.get(line).under.push(o);
            }
        });

        // 檢查每個盤口的套利機會
        for (const [line, odds] of lineGroups) {
            if (odds.over.length > 0 && odds.under.length > 0) {
                const bestOver = Math.max(...odds.over.map(o => o.odds));
                const bestUnder = Math.max(...odds.under.map(o => o.odds));

                const bestOverBookmaker = odds.over.find(o => o.odds === bestOver).bookmaker;
                const bestUnderBookmaker = odds.under.find(o => o.odds === bestUnder).bookmaker;

                const impliedProbOver = 1 / bestOver;
                const impliedProbUnder = 1 / bestUnder;
                const totalImpliedProb = impliedProbOver + impliedProbUnder;

                if (totalImpliedProb < 1 - this.config.oddsTolerance) {
                    const profitMargin = (1 - totalImpliedProb) / totalImpliedProb;
                    
                    const stakeOver = impliedProbOver / totalImpliedProb;
                    const stakeUnder = impliedProbUnder / totalImpliedProb;

                    this.metrics.arbitrageDetected.inc({ 
                        type: 'over-under', 
                        profit_range: this.getProfitRange(profitMargin) 
                    });
                    this.metrics.profitMargin.observe(profitMargin);

                    return {
                        type: 'over-under',
                        line,
                        profitMargin,
                        bookmakers: [bestOverBookmaker, bestUnderBookmaker],
                        selections: [`OVER_${line}`, `UNDER_${line}`],
                        odds: [bestOver, bestUnder],
                        stakes: [stakeOver, stakeUnder],
                        impliedProbability: totalImpliedProb,
                        expectedReturn: 1 / totalImpliedProb,
                        detectedAt: new Date().toISOString()
                    };
                }
            }
        }

        return null;
    }

    async publishArbitrageOpportunity(opportunity) {
        try {
            // 計算到期時間（考慮博彩公司延遲）
            const maxLatency = Math.max(
                ...opportunity.bookmakers.map(b => this.bookmakerLatency.get(b) || 100)
            );
            const expiresAt = new Date(Date.now() + maxLatency * 2);

            // 保存到資料庫
            const result = await this.db.query(`
                INSERT INTO arbitrage_opportunities (
                    event_id, opportunity_type, profit_margin,
                    bookmakers, selections, odds, stakes,
                    detected_at, expires_at, detection_latency_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            `, [
                opportunity.eventId,
                opportunity.type,
                opportunity.profitMargin,
                opportunity.bookmakers,
                opportunity.selections,
                opportunity.odds,
                opportunity.stakes,
                opportunity.detectedAt,
                expiresAt,
                opportunity.detectionLatency
            ]);

            opportunity.id = result.rows[0].id;
            opportunity.expiresAt = expiresAt;

            // 發布到 Kafka
            await this.producer.send({
                topic: 'arbitrage-opportunities',
                messages: [{
                    key: opportunity.eventId,
                    value: JSON.stringify(opportunity),
                    headers: {
                        'content-type': 'application/json',
                        'opportunity-type': opportunity.type
                    }
                }]
            });

            // 發布到 Redis（用於實時通知）
            await new Promise((resolve, reject) => {
                this.redis.publish('arbitrage:live', JSON.stringify(opportunity), (err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });

            // 觸發事件
            this.emit('arbitrage:detected', opportunity);

            logger.info('Published arbitrage opportunity:', {
                id: opportunity.id,
                type: opportunity.type,
                profitMargin: opportunity.profitMargin,
                eventId: opportunity.eventId
            });

        } catch (error) {
            logger.error('Error publishing arbitrage opportunity:', error);
        }
    }

    cleanupExpiredOdds() {
        const now = Date.now();
        let cleanedCount = 0;

        for (const [marketKey, marketOdds] of this.oddsCache) {
            for (const [selectionKey, odds] of marketOdds) {
                if (now > odds.expiresAt) {
                    marketOdds.delete(selectionKey);
                    cleanedCount++;
                }
            }
            
            if (marketOdds.size === 0) {
                this.oddsCache.delete(marketKey);
            }
        }

        if (cleanedCount > 0) {
            logger.debug(`Cleaned up ${cleanedCount} expired odds entries`);
        }
    }

    async saveStatistics() {
        try {
            const stats = {
                totalOpportunities: this.metrics.arbitrageDetected._getValue() || 0,
                avgProfitMargin: this.metrics.profitMargin.hashMap 
                    ? this.calculateAverageFromHistogram(this.metrics.profitMargin) 
                    : 0,
                avgDetectionLatency: this.metrics.detectionLatency.hashMap
                    ? this.calculateAverageFromHistogram(this.metrics.detectionLatency)
                    : 0,
                cacheSize: this.oddsCache.size,
                timestamp: new Date()
            };

            await this.db.query(`
                INSERT INTO arbitrage_statistics 
                (total_opportunities, avg_profit_margin, avg_detection_latency, 
                 cache_size, created_at)
                VALUES ($1, $2, $3, $4, $5)
            `, [
                stats.totalOpportunities,
                stats.avgProfitMargin,
                stats.avgDetectionLatency,
                stats.cacheSize,
                stats.timestamp
            ]);

        } catch (error) {
            logger.error('Error saving statistics:', error);
        }
    }

    calculateAverageFromHistogram(histogram) {
        let sum = 0;
        let count = 0;
        
        const values = histogram.hashMap || {};
        Object.keys(values).forEach(key => {
            const bucket = values[key];
            if (bucket && bucket.count) {
                sum += bucket.sum || 0;
                count += bucket.count;
            }
        });
        
        return count > 0 ? sum / count : 0;
    }

    getProfitRange(profitMargin) {
        if (profitMargin < 0.02) return 'low';
        if (profitMargin < 0.05) return 'medium';
        if (profitMargin < 0.10) return 'high';
        return 'very_high';
    }

    // 高級功能：跨市場套利檢測
    async detectCrossMarketArbitrage(eventId) {
        const opportunities = [];
        
        try {
            // 獲取相關市場數據
            const result = await this.db.query(`
                SELECT DISTINCT market_type, bookmaker, selection, odds_value
                FROM odds_history
                WHERE event_id = $1
                  AND timestamp > NOW() - INTERVAL '1 minute'
                  AND is_available = true
                ORDER BY market_type, selection, odds_value DESC
            `, [eventId]);

            // 分析亞洲盤與歐洲盤之間的套利
            const asianHandicap = result.rows.filter(r => r.market_type === 'AH');
            const european = result.rows.filter(r => r.market_type === '1X2');

            if (asianHandicap.length > 0 && european.length > 0) {
                // 實現複雜的跨市場套利邏輯
                const crossMarketOpp = this.calculateCrossMarketArbitrage(
                    asianHandicap, 
                    european
                );
                
                if (crossMarketOpp) {
                    opportunities.push(crossMarketOpp);
                }
            }

        } catch (error) {
            logger.error('Error detecting cross-market arbitrage:', error);
        }

        return opportunities;
    }

    calculateCrossMarketArbitrage(asianOdds, europeanOdds) {
        // 這裡實現複雜的跨市場套利計算邏輯
        // 例如：AH -0.5 與 1X2 市場的組合
        
        // 簡化示例
        const ahHome = asianOdds.find(o => o.selection === 'HOME_-0.5');
        const ahAway = asianOdds.find(o => o.selection === 'AWAY_+0.5');
        const euroHome = europeanOdds.find(o => o.selection === 'HOME');
        const euroDraw = europeanOdds.find(o => o.selection === 'DRAW');
        
        if (ahHome && euroDraw && ahAway) {
            // 計算組合套利機會
            const combinedProb = (1 / ahHome.odds_value) + 
                                (1 / euroDraw.odds_value) + 
                                (1 / ahAway.odds_value);
            
            if (combinedProb < 0.98) {
                return {
                    type: 'cross-market',
                    markets: ['AH', '1X2'],
                    profitMargin: (1 - combinedProb) / combinedProb,
                    combinations: [
                        { market: 'AH', selection: 'HOME_-0.5', odds: ahHome.odds_value },
                        { market: '1X2', selection: 'DRAW', odds: euroDraw.odds_value },
                        { market: 'AH', selection: 'AWAY_+0.5', odds: ahAway.odds_value }
                    ]
                };
            }
        }
        
        return null;
    }

    // 監控和健康檢查
    getHealthStatus() {
        return {
            status: this.isRunning ? 'healthy' : 'stopped',
            uptime: process.uptime(),
            metrics: {
                cacheSize: this.oddsCache.size,
                totalOpportunities: this.metrics.arbitrageDetected._getValue() || 0,
                kafkaConnected: this.consumer?.isConnected() || false,
                redisConnected: this.redis?.connected || false
            },
            config: this.config
        };
    }
}

// 輔助類：套利計算器
class ArbitrageCalculator {
    static calculateOptimalStakes(odds, totalStake = 1000) {
        const probabilities = odds.map(o => 1 / o);
        const totalProb = probabilities.reduce((sum, p) => sum + p, 0);
        
        if (totalProb >= 1) {
            return null; // 無套利機會
        }
        
        const stakes = probabilities.map(p => (p / totalProb) * totalStake);
        const returns = stakes.map((s, i) => s * odds[i]);
        const profit = Math.min(...returns) - totalStake;
        
        return {
            stakes,
            returns,
            profit,
            profitMargin: profit / totalStake,
            roi: (profit / totalStake) * 100
        };
    }
    
    static validateArbitrageOpportunity(opportunity, config) {
        // 驗證套利機會的有效性
        const { odds, stakes, profitMargin } = opportunity;
        
        // 檢查賠率是否合理
        if (odds.some(o => o < 1.01 || o > 1000)) {
            return { valid: false, reason: 'Invalid odds range' };
        }
        
        // 檢查投注比例
        const stakeSum = stakes.reduce((sum, s) => sum + s, 0);
        if (Math.abs(stakeSum - 1) > 0.001) {
            return { valid: false, reason: 'Invalid stake distribution' };
        }
        
        // 檢查利潤率
        if (profitMargin < config.minProfitThreshold) {
            return { valid: false, reason: 'Profit margin below threshold' };
        }
        
        // 重新計算驗證
        const recalc = this.calculateOptimalStakes(odds);
        if (!recalc || Math.abs(recalc.profitMargin - profitMargin) > 0.001) {
            return { valid: false, reason: 'Calculation mismatch' };
        }
        
        return { valid: true };
    }
}

// 監控服務
class ArbitrageMonitor {
    constructor(engine) {
        this.engine = engine;
        this.alerts = [];
    }
    
    startMonitoring() {
        // 監控檢測延遲
        setInterval(() => {
            const avgLatency = this.engine.calculateAverageFromHistogram(
                this.engine.metrics.detectionLatency
            );
            
            if (avgLatency > 500) {
                this.createAlert('HIGH_LATENCY', {
                    avgLatency,
                    threshold: 500
                });
            }
        }, 30000);
        
        // 監控套利機會數量
        setInterval(() => {
            const opportunities = this.engine.metrics.arbitrageDetected._getValue() || 0;
            
            if (opportunities === 0) {
                this.createAlert('NO_OPPORTUNITIES', {
                    duration: '30 minutes'
                });
            }
        }, 1800000); // 30分鐘
    }
    
    createAlert(type, data) {
        const alert = {
            type,
            data,
            timestamp: new Date(),
            resolved: false
        };
        
        this.alerts.push(alert);
        logger.warn('Arbitrage monitor alert:', alert);
        
        // 發送通知（可以整合 email、Slack 等）
        this.sendNotification(alert);
    }
    
    async sendNotification(alert) {
        // 實現通知邏輯
        try {
            // 例如：發送到 Slack
            // await slackClient.send(alert);
        } catch (error) {
            logger.error('Failed to send notification:', error);
        }
    }
}

// 導出模組
module.exports = {
    ArbitrageDetectionEngine,
    ArbitrageCalculator,
    ArbitrageMonitor
};

// 如果直接運行此文件，啟動獨立的套利檢測服務
if (require.main === module) {
    const engine = new ArbitrageDetectionEngine();
    const monitor = new ArbitrageMonitor(engine);
    
    async function start() {
        try {
            await engine.initialize();
            await engine.start();
            monitor.startMonitoring();
            
            logger.info('Arbitrage detection service started');
            
            // 優雅關機
            process.on('SIGTERM', async () => {
                logger.info('Shutting down arbitrage detection service');
                await engine.stop();
                process.exit(0);
            });
            
        } catch (error) {
            logger.error('Failed to start arbitrage detection service:', error);
            process.exit(1);
        }
    }
    
    start();
}