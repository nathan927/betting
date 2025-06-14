// monitoring-alerting-system.js - 生產級監控告警系統
const promClient = require('prom-client');
const express = require('express');
const winston = require('winston');
const nodemailer = require('nodemailer');
const { WebClient } = require('@slack/web-api');
const EventEmitter = require('events');
const cron = require('node-cron');
const os = require('os');
const { Pool } = require('pg');
const Redis = require('redis');

// 配置日誌
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/monitoring.log' })
    ]
});

// 監控指標定義
class MetricsDefinition {
    constructor() {
        this.register = new promClient.Registry();
        
        // 系統指標
        this.systemMetrics = {
            cpuUsage: new promClient.Gauge({
                name: 'system_cpu_usage_percent',
                help: 'CPU usage percentage',
                labelNames: ['core']
            }),
            
            memoryUsage: new promClient.Gauge({
                name: 'system_memory_usage_bytes',
                help: 'Memory usage in bytes',
                labelNames: ['type']
            }),
            
            diskUsage: new promClient.Gauge({
                name: 'system_disk_usage_bytes',
                help: 'Disk usage in bytes',
                labelNames: ['mount']
            })
        };
        
        // 應用指標
        this.appMetrics = {
            httpRequestTotal: new promClient.Counter({
                name: 'http_requests_total',
                help: 'Total number of HTTP requests',
                labelNames: ['method', 'endpoint', 'status']
            }),
            
            httpRequestDuration: new promClient.Histogram({
                name: 'http_request_duration_seconds',
                help: 'HTTP request latencies',
                labelNames: ['method', 'endpoint', 'status'],
                buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]
            }),
            
            activeConnections: new promClient.Gauge({
                name: 'active_connections',
                help: 'Number of active connections',
                labelNames: ['type']
            }),
            
            errorRate: new promClient.Counter({
                name: 'errors_total',
                help: 'Total number of errors',
                labelNames: ['type', 'severity']
            })
        };
        
        // 業務指標
        this.businessMetrics = {
            betsPlaced: new promClient.Counter({
                name: 'bets_placed_total',
                help: 'Total number of bets placed',
                labelNames: ['sport', 'market_type']
            }),
            
            betAmount: new promClient.Histogram({
                name: 'bet_amount_usd',
                help: 'Bet amounts in USD',
                labelNames: ['sport'],
                buckets: [1, 5, 10, 50, 100, 500, 1000, 5000]
            }),
            
            arbitrageDetected: new promClient.Counter({
                name: 'arbitrage_opportunities_detected',
                help: 'Number of arbitrage opportunities detected',
                labelNames: ['type', 'profit_range']
            }),
            
            modelAccuracy: new promClient.Gauge({
                name: 'ml_model_accuracy',
                help: 'Machine learning model accuracy',
                labelNames: ['model', 'sport']
            }),
            
            cacheHitRate: new promClient.Gauge({
                name: 'cache_hit_rate',
                help: 'Cache hit rate percentage',
                labelNames: ['cache_type']
            }),
            
            scraperSuccess: new promClient.Gauge({
                name: 'scraper_success_rate',
                help: 'Web scraper success rate',
                labelNames: ['bookmaker']
            })
        };
        
        // 註冊所有指標
        Object.values(this.systemMetrics).forEach(metric => this.register.registerMetric(metric));
        Object.values(this.appMetrics).forEach(metric => this.register.registerMetric(metric));
        Object.values(this.businessMetrics).forEach(metric => this.register.registerMetric(metric));
        
        // 預設指標
        promClient.collectDefaultMetrics({ register: this.register });
    }
}

// 告警規則引擎
class AlertRuleEngine extends EventEmitter {
    constructor() {
        super();
        this.rules = new Map();
        this.alertHistory = new Map();
        this.cooldownPeriods = new Map();
        
        this.setupDefaultRules();
    }
    
    setupDefaultRules() {
        // CPU 使用率告警
        this.addRule({
            id: 'high_cpu_usage',
            name: 'High CPU Usage',
            metric: 'system_cpu_usage_percent',
            condition: (value) => value > 80,
            severity: 'warning',
            cooldown: 300, // 5分鐘
            message: (value) => `CPU usage is at ${value.toFixed(1)}%`
        });
        
        // 內存使用告警
        this.addRule({
            id: 'high_memory_usage',
            name: 'High Memory Usage',
            metric: 'system_memory_usage_percent',
            condition: (value) => value > 85,
            severity: 'warning',
            cooldown: 300,
            message: (value) => `Memory usage is at ${value.toFixed(1)}%`
        });
        
        // API 錯誤率告警
        this.addRule({
            id: 'high_error_rate',
            name: 'High Error Rate',
            metric: 'error_rate_5m',
            condition: (value) => value > 0.05, // 5%
            severity: 'critical',
            cooldown: 600,
            message: (value) => `Error rate is at ${(value * 100).toFixed(2)}%`
        });
        
        // 響應時間告警
        this.addRule({
            id: 'slow_response_time',
            name: 'Slow Response Time',
            metric: 'http_request_duration_p95',
            condition: (value) => value > 2, // 2秒
            severity: 'warning',
            cooldown: 300,
            message: (value) => `95th percentile response time is ${value.toFixed(2)}s`
        });
        
        // 套利檢測延遲告警
        this.addRule({
            id: 'arbitrage_detection_slow',
            name: 'Arbitrage Detection Slow',
            metric: 'arbitrage_detection_latency_ms',
            condition: (value) => value > 500,
            severity: 'critical',
            cooldown: 180,
            message: (value) => `Arbitrage detection latency is ${value}ms`
        });
        
        // 緩存命中率告警
        this.addRule({
            id: 'low_cache_hit_rate',
            name: 'Low Cache Hit Rate',
            metric: 'cache_hit_rate',
            condition: (value) => value < 0.8, // 80%
            severity: 'warning',
            cooldown: 600,
            message: (value) => `Cache hit rate is ${(value * 100).toFixed(1)}%`
        });
        
        // 數據庫連接池告警
        this.addRule({
            id: 'db_connection_pool_exhausted',
            name: 'Database Connection Pool Exhausted',
            metric: 'db_connection_pool_available',
            condition: (value) => value < 5,
            severity: 'critical',
            cooldown: 60,
            message: (value) => `Only ${value} database connections available`
        });
        
        // 爬蟲失敗率告警
        this.addRule({
            id: 'scraper_high_failure_rate',
            name: 'Scraper High Failure Rate',
            metric: 'scraper_failure_rate',
            condition: (value, labels) => value > 0.2 && labels.bookmaker,
            severity: 'warning',
            cooldown: 900,
            message: (value, labels) => `${labels.bookmaker} scraper failure rate is ${(value * 100).toFixed(1)}%`
        });
    }
    
    addRule(rule) {
        this.rules.set(rule.id, {
            ...rule,
            lastTriggered: null,
            isActive: true
        });
    }
    
    async evaluate(metrics) {
        const triggeredAlerts = [];
        
        for (const [ruleId, rule] of this.rules) {
            if (!rule.isActive) continue;
            
            try {
                const metricValue = this.getMetricValue(metrics, rule.metric);
                const labels = this.getMetricLabels(metrics, rule.metric);
                
                if (metricValue !== null && rule.condition(metricValue, labels)) {
                    if (this.shouldTriggerAlert(ruleId)) {
                        const alert = {
                            ruleId,
                            ruleName: rule.name,
                            severity: rule.severity,
                            message: rule.message(metricValue, labels),
                            value: metricValue,
                            labels,
                            timestamp: new Date()
                        };
                        
                        triggeredAlerts.push(alert);
                        this.recordAlertTriggered(ruleId);
                        this.emit('alert', alert);
                    }
                }
            } catch (error) {
                logger.error(`Error evaluating rule ${ruleId}:`, error);
            }
        }
        
        return triggeredAlerts;
    }
    
    shouldTriggerAlert(ruleId) {
        const rule = this.rules.get(ruleId);
        if (!rule) return false;
        
        const now = Date.now();
        const cooldownKey = `${ruleId}:cooldown`;
        
        // 檢查冷卻期
        if (this.cooldownPeriods.has(cooldownKey)) {
            const cooldownEnd = this.cooldownPeriods.get(cooldownKey);
            if (now < cooldownEnd) {
                return false;
            }
        }
        
        return true;
    }
    
    recordAlertTriggered(ruleId) {
        const rule = this.rules.get(ruleId);
        if (!rule) return;
        
        const now = Date.now();
        rule.lastTriggered = now;
        
        // 設置冷卻期
        const cooldownKey = `${ruleId}:cooldown`;
        const cooldownEnd = now + (rule.cooldown * 1000);
        this.cooldownPeriods.set(cooldownKey, cooldownEnd);
        
        // 記錄歷史
        if (!this.alertHistory.has(ruleId)) {
            this.alertHistory.set(ruleId, []);
        }
        this.alertHistory.get(ruleId).push(now);
    }
    
    getMetricValue(metrics, metricName) {
        // 實現從 metrics 對象中提取指定指標的值
        return metrics[metricName]?.value || null;
    }
    
    getMetricLabels(metrics, metricName) {
        // 實現從 metrics 對象中提取指定指標的標籤
        return metrics[metricName]?.labels || {};
    }
}

// 通知管理器
class NotificationManager {
    constructor(config) {
        this.config = config;
        this.channels = new Map();
        
        this.setupChannels();
    }
    
    setupChannels() {
        // Email 通道
        if (this.config.email?.enabled) {
            this.channels.set('email', new EmailNotifier(this.config.email));
        }
        
        // Slack 通道
        if (this.config.slack?.enabled) {
            this.channels.set('slack', new SlackNotifier(this.config.slack));
        }
        
        // SMS 通道
        if (this.config.sms?.enabled) {
            this.channels.set('sms', new SMSNotifier(this.config.sms));
        }
        
        // Webhook 通道
        if (this.config.webhook?.enabled) {
            this.channels.set('webhook', new WebhookNotifier(this.config.webhook));
        }
    }
    
    async sendAlert(alert) {
        const promises = [];
        
        // 根據嚴重程度決定使用哪些通道
        const channelsToUse = this.getChannelsForSeverity(alert.severity);
        
        for (const channelName of channelsToUse) {
            const channel = this.channels.get(channelName);
            if (channel) {
                promises.push(
                    channel.send(alert)
                        .catch(error => {
                            logger.error(`Failed to send alert via ${channelName}:`, error);
                        })
                );
            }
        }
        
        await Promise.allSettled(promises);
    }
    
    getChannelsForSeverity(severity) {
        switch (severity) {
            case 'critical':
                return ['email', 'slack', 'sms'];
            case 'warning':
                return ['email', 'slack'];
            case 'info':
                return ['slack'];
            default:
                return ['slack'];
        }
    }
}

// Email 通知器
class EmailNotifier {
    constructor(config) {
        this.transporter = nodemailer.createTransport({
            host: config.smtp.host,
            port: config.smtp.port,
            secure: config.smtp.secure,
            auth: {
                user: config.smtp.user,
                pass: config.smtp.pass
            }
        });
        
        this.recipients = config.recipients;
    }
    
    async send(alert) {
        const mailOptions = {
            from: '"Betting System Monitor" <monitor@betting-system.com>',
            to: this.recipients.join(', '),
            subject: `[${alert.severity.toUpperCase()}] ${alert.ruleName}`,
            html: this.formatAlertEmail(alert)
        };
        
        await this.transporter.sendMail(mailOptions);
        logger.info(`Email alert sent for ${alert.ruleId}`);
    }
    
    formatAlertEmail(alert) {
        return `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: ${this.getSeverityColor(alert.severity)}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">${alert.ruleName}</h2>
                    <p style="margin: 5px 0 0 0;">Severity: ${alert.severity.toUpperCase()}</p>
                </div>
                <div style="background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none;">
                    <p><strong>Alert Message:</strong></p>
                    <p style="font-size: 16px; color: #333;">${alert.message}</p>
                    
                    <p><strong>Details:</strong></p>
                    <ul style="color: #666;">
                        <li>Time: ${alert.timestamp.toISOString()}</li>
                        <li>Value: ${alert.value}</li>
                        ${alert.labels ? `<li>Labels: ${JSON.stringify(alert.labels)}</li>` : ''}
                    </ul>
                    
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                    
                    <p style="color: #999; font-size: 12px;">
                        This is an automated alert from the Betting System Monitoring Service.
                    </p>
                </div>
            </div>
        `;
    }
    
    getSeverityColor(severity) {
        switch (severity) {
            case 'critical': return '#dc3545';
            case 'warning': return '#ffc107';
            case 'info': return '#17a2b8';
            default: return '#6c757d';
        }
    }
}

// Slack 通知器
class SlackNotifier {
    constructor(config) {
        this.client = new WebClient(config.token);
        this.channel = config.channel;
    }
    
    async send(alert) {
        const attachment = {
            color: this.getSeverityColor(alert.severity),
            title: alert.ruleName,
            text: alert.message,
            fields: [
                {
                    title: 'Severity',
                    value: alert.severity.toUpperCase(),
                    short: true
                },
                {
                    title: 'Time',
                    value: alert.timestamp.toLocaleString(),
                    short: true
                },
                {
                    title: 'Value',
                    value: String(alert.value),
                    short: true
                }
            ],
            footer: 'Betting System Monitor',
            ts: Math.floor(alert.timestamp.getTime() / 1000)
        };
        
        if (alert.labels && Object.keys(alert.labels).length > 0) {
            attachment.fields.push({
                title: 'Labels',
                value: JSON.stringify(alert.labels),
                short: false
            });
        }
        
        await this.client.chat.postMessage({
            channel: this.channel,
            text: `Alert: ${alert.ruleName}`,
            attachments: [attachment]
        });
        
        logger.info(`Slack alert sent for ${alert.ruleId}`);
    }
    
    getSeverityColor(severity) {
        switch (severity) {
            case 'critical': return '#dc3545';
            case 'warning': return '#ffc107';
            case 'info': return '#17a2b8';
            default: return '#6c757d';
        }
    }
}

// 監控服務主類
class MonitoringService {
    constructor(config) {
        this.config = config;
        this.metrics = new MetricsDefinition();
        this.alertEngine = new AlertRuleEngine();
        this.notificationManager = new NotificationManager(config.notifications);
        this.db = null;
        this.redis = null;
        
        this.setupAlertHandlers();
    }
    
    async initialize() {
        // 初始化數據庫連接
        this.db = new Pool({
            host: this.config.database.host,
            port: this.config.database.port,
            database: this.config.database.name,
            user: this.config.database.user,
            password: this.config.database.password
        });
        
        // 初始化 Redis
        this.redis = Redis.createClient({
            host: this.config.redis.host,
            port: this.config.redis.port,
            password: this.config.redis.password
        });
        
        // 啟動指標收集
        this.startMetricsCollection();
        
        // 啟動告警評估
        this.startAlertEvaluation();
        
        // 啟動 HTTP 服務器（用於 Prometheus）
        this.startMetricsServer();
        
        logger.info('Monitoring service initialized');
    }
    
    setupAlertHandlers() {
        this.alertEngine.on('alert', async (alert) => {
            logger.warn('Alert triggered:', alert);
            
            // 發送通知
            await this.notificationManager.sendAlert(alert);
            
            // 記錄到數據庫
            await this.recordAlert(alert);
        });
    }
    
    startMetricsCollection() {
        // 系統指標收集（每10秒）
        setInterval(() => {
            this.collectSystemMetrics();
        }, 10000);
        
        // 應用指標收集（每30秒）
        setInterval(() => {
            this.collectApplicationMetrics();
        }, 30000);
        
        // 業務指標收集（每分鐘）
        setInterval(() => {
            this.collectBusinessMetrics();
        }, 60000);
    }
    
    async collectSystemMetrics() {
        // CPU 使用率
        const cpus = os.cpus();
        cpus.forEach((cpu, index) => {
            const total = Object.values(cpu.times).reduce((a, b) => a + b);
            const idle = cpu.times.idle;
            const usage = 100 - (idle / total * 100);
            
            this.metrics.systemMetrics.cpuUsage
                .labels({ core: String(index) })
                .set(usage);
        });
        
        // 內存使用
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const usedMem = totalMem - freeMem;
        
        this.metrics.systemMetrics.memoryUsage
            .labels({ type: 'used' })
            .set(usedMem);
        
        this.metrics.systemMetrics.memoryUsage
            .labels({ type: 'free' })
            .set(freeMem);
    }
    
    async collectApplicationMetrics() {
        try {
            // 數據庫連接池狀態
            const poolStats = await this.db.query(`
                SELECT 
                    COUNT(*) FILTER (WHERE state = 'active') as active,
                    COUNT(*) FILTER (WHERE state = 'idle') as idle,
                    COUNT(*) as total
                FROM pg_stat_activity
                WHERE datname = current_database()
            `);
            
            const stats = poolStats.rows[0];
            this.metrics.appMetrics.activeConnections
                .labels({ type: 'database' })
                .set(parseInt(stats.active));
            
            // Redis 連接狀態
            const redisInfo = await new Promise((resolve, reject) => {
                this.redis.info((err, info) => {
                    if (err) reject(err);
                    else resolve(info);
                });
            });
            
            const connectedClients = redisInfo.match(/connected_clients:(\d+)/);
            if (connectedClients) {
                this.metrics.appMetrics.activeConnections
                    .labels({ type: 'redis' })
                    .set(parseInt(connectedClients[1]));
            }
            
        } catch (error) {
            logger.error('Error collecting application metrics:', error);
        }
    }
    
    async collectBusinessMetrics() {
        try {
            // 緩存命中率
            const cacheStats = await this.getCacheStats();
            this.metrics.businessMetrics.cacheHitRate
                .labels({ cache_type: 'redis' })
                .set(cacheStats.hitRate);
            
            // 模型準確率
            const modelStats = await this.getModelStats();
            for (const [model, accuracy] of Object.entries(modelStats)) {
                this.metrics.businessMetrics.modelAccuracy
                    .labels({ model, sport: 'horse_racing' })
                    .set(accuracy);
            }
            
            // 爬蟲成功率
            const scraperStats = await this.getScraperStats();
            for (const [bookmaker, successRate] of Object.entries(scraperStats)) {
                this.metrics.businessMetrics.scraperSuccess
                    .labels({ bookmaker })
                    .set(successRate);
            }
            
        } catch (error) {
            logger.error('Error collecting business metrics:', error);
        }
    }
    
    async getCacheStats() {
        // 實現緩存統計邏輯
        const info = await new Promise((resolve, reject) => {
            this.redis.info('stats', (err, data) => {
                if (err) reject(err);
                else resolve(data);
            });
        });
        
        const hits = parseInt(info.match(/keyspace_hits:(\d+)/)?.[1] || 0);
        const misses = parseInt(info.match(/keyspace_misses:(\d+)/)?.[1] || 0);
        const total = hits + misses;
        
        return {
            hits,
            misses,
            hitRate: total > 0 ? hits / total : 0
        };
    }
    
    async getModelStats() {
        // 從數據庫獲取模型統計
        const result = await this.db.query(`
            SELECT 
                model_name,
                accuracy
            FROM ml_models
            WHERE is_active = true
        `);
        
        const stats = {};
        result.rows.forEach(row => {
            stats[row.model_name] = parseFloat(row.accuracy);
        });
        
        return stats;
    }
    
    async getScraperStats() {
        // 從數據庫獲取爬蟲統計
        const result = await this.db.query(`
            SELECT 
                bookmaker,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
            FROM scraper_logs
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY bookmaker
        `);
        
        const stats = {};
        result.rows.forEach(row => {
            stats[row.bookmaker] = parseFloat(row.success_rate);
        });
        
        return stats;
    }
    
    startAlertEvaluation() {
        // 每分鐘評估一次告警規則
        setInterval(async () => {
            try {
                const currentMetrics = await this.getCurrentMetrics();
                await this.alertEngine.evaluate(currentMetrics);
            } catch (error) {
                logger.error('Error evaluating alerts:', error);
            }
        }, 60000);
    }
    
    async getCurrentMetrics() {
        // 收集當前所有指標值
        const metrics = {};
        
        // 從 Prometheus 註冊表獲取指標
        const metricFamilies = this.metrics.register.getMetricsAsArray();
        
        for (const family of metricFamilies) {
            const values = await family.get();
            metrics[family.name] = values;
        }
        
        return metrics;
    }
    
    async recordAlert(alert) {
        try {
            await this.db.query(`
                INSERT INTO alert_history 
                (rule_id, rule_name, severity, message, value, labels, triggered_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            `, [
                alert.ruleId,
                alert.ruleName,
                alert.severity,
                alert.message,
                alert.value,
                JSON.stringify(alert.labels),
                alert.timestamp
            ]);
        } catch (error) {
            logger.error('Error recording alert:', error);
        }
    }
    
    startMetricsServer() {
        const app = express();
        
        // Prometheus 端點
        app.get('/metrics', async (req, res) => {
            res.set('Content-Type', this.metrics.register.contentType);
            const metrics = await this.metrics.register.metrics();
            res.end(metrics);
        });
        
        // 健康檢查
        app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                uptime: process.uptime(),
                timestamp: new Date().toISOString()
            });
        });
        
        // 告警狀態
        app.get('/alerts', (req, res) => {
            const activeAlerts = [];
            
            for (const [ruleId, rule] of this.alertEngine.rules) {
                if (rule.lastTriggered) {
                    activeAlerts.push({
                        ruleId,
                        ruleName: rule.name,
                        severity: rule.severity,
                        lastTriggered: new Date(rule.lastTriggered).toISOString()
                    });
                }
            }
            
            res.json({ activeAlerts });
        });
        
        const port = this.config.metricsPort || 9090;
        app.listen(port, () => {
            logger.info(`Metrics server listening on port ${port}`);
        });
    }
}

// 配置
const config = {
    metricsPort: 9090,
    database: {
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        name: process.env.DB_NAME || 'betting_db',
        user: process.env.DB_USER || 'betting_user',
        password: process.env.DB_PASSWORD
    },
    redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || 6379,
        password: process.env.REDIS_PASSWORD
    },
    notifications: {
        email: {
            enabled: true,
            smtp: {
                host: 'smtp.gmail.com',
                port: 587,
                secure: false,
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS
            },
            recipients: ['admin@betting-system.com', 'ops@betting-system.com']
        },
        slack: {
            enabled: true,
            token: process.env.SLACK_TOKEN,
            channel: '#betting-alerts'
        },
        sms: {
            enabled: false,
            // Twilio 配置
        },
        webhook: {
            enabled: true,
            url: process.env.WEBHOOK_URL
        }
    }
};

// 主程序
async function main() {
    const monitoringService = new MonitoringService(config);
    
    try {
        await monitoringService.initialize();
        logger.info('Monitoring service started successfully');
        
        // 優雅關機
        process.on('SIGTERM', async () => {
            logger.info('Shutting down monitoring service');
            await monitoringService.db.end();
            monitoringService.redis.quit();
            process.exit(0);
        });
        
    } catch (error) {
        logger.error('Failed to start monitoring service:', error);
        process.exit(1);
    }
}

// 導出模組
module.exports = {
    MonitoringService,
    AlertRuleEngine,
    NotificationManager,
    MetricsDefinition
};

// 如果直接運行此文件
if (require.main === module) {
    main();
}