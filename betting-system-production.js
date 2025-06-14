// server.js - 生產級博彩系統主服務器
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const { Pool } = require('pg');
const Redis = require('redis');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const WebSocket = require('ws');
const { Kafka } = require('kafkajs');
const promClient = require('prom-client');
const winston = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');

// 配置日誌系統
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.splat(),
        winston.format.json()
    ),
    defaultMeta: { service: 'betting-system' },
    transports: [
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        }),
        new DailyRotateFile({
            filename: 'logs/application-%DATE%.log',
            datePattern: 'YYYY-MM-DD',
            zippedArchive: true,
            maxSize: '20m',
            maxFiles: '14d'
        }),
        new DailyRotateFile({
            filename: 'logs/error-%DATE%.log',
            datePattern: 'YYYY-MM-DD',
            zippedArchive: true,
            maxSize: '20m',
            maxFiles: '14d',
            level: 'error'
        })
    ]
});

// 資料庫管理器
class DatabaseManager {
    constructor() {
        this.pool = new Pool({
            host: process.env.DB_HOST || 'localhost',
            port: process.env.DB_PORT || 5432,
            database: process.env.DB_NAME || 'betting_db',
            user: process.env.DB_USER || 'betting_user',
            password: process.env.DB_PASSWORD,
            max: 50,
            idleTimeoutMillis: 30000,
            connectionTimeoutMillis: 2000,
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        });

        this.redis = Redis.createClient({
            host: process.env.REDIS_HOST || 'localhost',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD,
            retry_strategy: (options) => {
                if (options.error && options.error.code === 'ECONNREFUSED') {
                    return new Error('Redis server refused connection');
                }
                if (options.total_retry_time > 1000 * 60 * 60) {
                    return new Error('Redis retry time exhausted');
                }
                if (options.attempt > 10) {
                    return undefined;
                }
                return Math.min(options.attempt * 100, 3000);
            }
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.pool.on('error', (err) => {
            logger.error('PostgreSQL pool error:', err);
        });

        this.redis.on('error', (err) => {
            logger.error('Redis client error:', err);
        });

        this.redis.on('connect', () => {
            logger.info('Redis client connected');
        });

        this.redis.on('ready', () => {
            logger.info('Redis client ready');
        });
    }

    async initialize() {
        try {
            const client = await this.pool.connect();
            await client.query('SELECT NOW()');
            client.release();
            logger.info('PostgreSQL connection established');

            await new Promise((resolve, reject) => {
                this.redis.ping((err, result) => {
                    if (err) reject(err);
                    else resolve(result);
                });
            });
            logger.info('Redis connection established');

            return true;
        } catch (error) {
            logger.error('Database initialization failed:', error);
            throw error;
        }
    }

    async query(text, params) {
        const start = Date.now();
        try {
            const res = await this.pool.query(text, params);
            const duration = Date.now() - start;
            logger.debug('Executed query', { text, duration, rows: res.rowCount });
            return res;
        } catch (error) {
            logger.error('Database query error:', { text, error: error.message });
            throw error;
        }
    }

    async getClient() {
        const client = await this.pool.connect();
        const query = client.query.bind(client);
        const release = () => {
            client.release();
        };
        
        // 設置超時自動釋放
        const timeout = setTimeout(() => {
            logger.error('Client checkout timeout - forcing release');
            client.release();
        }, 5000);

        client.query = (...args) => {
            clearTimeout(timeout);
            return query(...args);
        };

        client.release = () => {
            clearTimeout(timeout);
            return release();
        };

        return client;
    }

    async transaction(callback) {
        const client = await this.getClient();
        try {
            await client.query('BEGIN');
            const result = await callback(client);
            await client.query('COMMIT');
            return result;
        } catch (error) {
            await client.query('ROLLBACK');
            throw error;
        } finally {
            client.release();
        }
    }
}

// 安全管理器
class SecurityManager {
    constructor() {
        this.jwtSecret = process.env.JWT_SECRET || 'default-secret-change-this';
        this.jwtRefreshSecret = process.env.JWT_REFRESH_SECRET || 'default-refresh-secret';
        this.saltRounds = 12;
        
        // 速率限制器
        this.authLimiter = rateLimit({
            windowMs: 15 * 60 * 1000,
            max: 5,
            message: 'Too many authentication attempts',
            standardHeaders: true,
            legacyHeaders: false,
            skipSuccessfulRequests: true,
            handler: (req, res) => {
                logger.warn('Rate limit exceeded:', { ip: req.ip, path: req.path });
                res.status(429).json({ error: 'Too many requests' });
            }
        });

        this.betLimiter = rateLimit({
            windowMs: 60 * 1000,
            max: 10,
            message: 'Too many betting requests',
            keyGenerator: (req) => req.user?.id || req.ip,
            skip: (req) => req.user?.role === 'vip'
        });

        this.apiLimiter = rateLimit({
            windowMs: 15 * 60 * 1000,
            max: 1000,
            message: 'API rate limit exceeded'
        });
    }

    async hashPassword(password) {
        if (!password || password.length < 8) {
            throw new Error('Password must be at least 8 characters');
        }
        return await bcrypt.hash(password, this.saltRounds);
    }

    async verifyPassword(password, hashedPassword) {
        return await bcrypt.compare(password, hashedPassword);
    }

    generateTokens(user) {
        const accessToken = jwt.sign(
            { 
                userId: user.id, 
                username: user.username,
                role: user.role,
                type: 'access'
            },
            this.jwtSecret,
            { 
                expiresIn: '15m',
                issuer: 'betting-system',
                audience: 'betting-api'
            }
        );

        const refreshToken = jwt.sign(
            { 
                userId: user.id,
                type: 'refresh'
            },
            this.jwtRefreshSecret,
            { 
                expiresIn: '7d',
                issuer: 'betting-system'
            }
        );

        return { accessToken, refreshToken };
    }

    verifyAccessToken(token) {
        try {
            const decoded = jwt.verify(token, this.jwtSecret, {
                issuer: 'betting-system',
                audience: 'betting-api'
            });
            
            if (decoded.type !== 'access') {
                throw new Error('Invalid token type');
            }
            
            return decoded;
        } catch (error) {
            logger.warn('Token verification failed:', error.message);
            throw new Error('Invalid token');
        }
    }

    verifyRefreshToken(token) {
        try {
            const decoded = jwt.verify(token, this.jwtRefreshSecret, {
                issuer: 'betting-system'
            });
            
            if (decoded.type !== 'refresh') {
                throw new Error('Invalid token type');
            }
            
            return decoded;
        } catch (error) {
            throw new Error('Invalid refresh token');
        }
    }

    // 中間件：驗證存取令牌
    authenticateToken(req, res, next) {
        const authHeader = req.headers['authorization'];
        const token = authHeader && authHeader.split(' ')[1];

        if (!token) {
            return res.status(401).json({ error: 'Access token required' });
        }

        try {
            const decoded = this.verifyAccessToken(token);
            req.user = decoded;
            next();
        } catch (error) {
            return res.status(403).json({ error: 'Invalid or expired token' });
        }
    }

    // 中間件：檢查用戶角色
    requireRole(...roles) {
        return (req, res, next) => {
            if (!req.user) {
                return res.status(401).json({ error: 'Authentication required' });
            }

            if (!roles.includes(req.user.role)) {
                logger.warn('Unauthorized access attempt:', { 
                    userId: req.user.userId, 
                    role: req.user.role, 
                    requiredRoles: roles 
                });
                return res.status(403).json({ error: 'Insufficient permissions' });
            }

            next();
        };
    }
}

// 緩存管理器
class CacheManager {
    constructor(redis) {
        this.redis = redis;
        this.defaultTTL = 300; // 5 minutes
        this.metrics = {
            hits: 0,
            misses: 0,
            errors: 0
        };
    }

    async get(key) {
        try {
            const value = await new Promise((resolve, reject) => {
                this.redis.get(key, (err, data) => {
                    if (err) reject(err);
                    else resolve(data);
                });
            });

            if (value) {
                this.metrics.hits++;
                return JSON.parse(value);
            } else {
                this.metrics.misses++;
                return null;
            }
        } catch (error) {
            this.metrics.errors++;
            logger.error('Cache get error:', { key, error: error.message });
            return null;
        }
    }

    async set(key, value, ttl = this.defaultTTL) {
        try {
            const serialized = JSON.stringify(value);
            await new Promise((resolve, reject) => {
                this.redis.setex(key, ttl, serialized, (err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            return true;
        } catch (error) {
            this.metrics.errors++;
            logger.error('Cache set error:', { key, error: error.message });
            return false;
        }
    }

    async del(key) {
        try {
            await new Promise((resolve, reject) => {
                this.redis.del(key, (err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            return true;
        } catch (error) {
            logger.error('Cache delete error:', { key, error: error.message });
            return false;
        }
    }

    async flush() {
        try {
            await new Promise((resolve, reject) => {
                this.redis.flushdb((err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            return true;
        } catch (error) {
            logger.error('Cache flush error:', error.message);
            return false;
        }
    }

    getHitRate() {
        const total = this.metrics.hits + this.metrics.misses;
        return total > 0 ? this.metrics.hits / total : 0;
    }

    getMetrics() {
        return {
            ...this.metrics,
            hitRate: this.getHitRate()
        };
    }
}

// 指標收集器
class MetricsCollector {
    constructor() {
        this.register = new promClient.Registry();
        
        // HTTP 請求持續時間
        this.httpRequestDuration = new promClient.Histogram({
            name: 'http_request_duration_seconds',
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['method', 'route', 'status_code'],
            buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
        });

        // 活躍連接數
        this.activeConnections = new promClient.Gauge({
            name: 'active_connections',
            help: 'Number of active connections',
            labelNames: ['type']
        });

        // 業務指標
        this.businessMetrics = {
            totalBets: new promClient.Counter({
                name: 'total_bets',
                help: 'Total number of bets placed',
                labelNames: ['sport', 'bet_type']
            }),
            betAmount: new promClient.Histogram({
                name: 'bet_amount',
                help: 'Distribution of bet amounts',
                labelNames: ['currency'],
                buckets: [10, 50, 100, 500, 1000, 5000, 10000]
            }),
            userRegistrations: new promClient.Counter({
                name: 'user_registrations_total',
                help: 'Total number of user registrations'
            })
        };

        // 註冊指標
        this.register.registerMetric(this.httpRequestDuration);
        this.register.registerMetric(this.activeConnections);
        Object.values(this.businessMetrics).forEach(metric => {
            this.register.registerMetric(metric);
        });

        // 預設指標
        promClient.collectDefaultMetrics({
            register: this.register,
            prefix: 'betting_system_'
        });
    }

    recordHttpRequest(method, route, statusCode, duration) {
        this.httpRequestDuration
            .labels(method, route, statusCode.toString())
            .observe(duration);
    }

    recordBet(sport, betType, amount, currency = 'USD') {
        this.businessMetrics.totalBets.labels(sport, betType).inc();
        this.businessMetrics.betAmount.labels(currency).observe(amount);
    }

    recordUserRegistration() {
        this.businessMetrics.userRegistrations.inc();
    }

    updateActiveConnections(type, delta) {
        this.activeConnections.labels(type).inc(delta);
    }

    getMetrics() {
        return this.register.metrics();
    }
}

// 主應用程式類
class BettingSystemApp {
    constructor() {
        this.app = express();
        this.db = new DatabaseManager();
        this.security = new SecurityManager();
        this.metrics = new MetricsCollector();
        this.cache = null; // 將在初始化後設置
        this.wsServer = null;
        this.kafka = null;
    }

    async initialize() {
        try {
            // 初始化資料庫
            await this.db.initialize();
            
            // 設置緩存管理器
            this.cache = new CacheManager(this.db.redis);
            
            // 設置中間件
            this.setupMiddleware();
            
            // 設置路由
            this.setupRoutes();
            
            // 設置錯誤處理
            this.setupErrorHandling();
            
            // 初始化 Kafka
            await this.initializeKafka();
            
            // 初始化 WebSocket
            this.initializeWebSocket();
            
            logger.info('Application initialized successfully');
        } catch (error) {
            logger.error('Application initialization failed:', error);
            throw error;
        }
    }

    setupMiddleware() {
        // 請求日誌
        this.app.use((req, res, next) => {
            const start = Date.now();
            res.on('finish', () => {
                const duration = (Date.now() - start) / 1000;
                this.metrics.recordHttpRequest(
                    req.method,
                    req.route?.path || req.path,
                    res.statusCode,
                    duration
                );
                
                logger.info('HTTP Request', {
                    method: req.method,
                    path: req.path,
                    statusCode: res.statusCode,
                    duration: `${duration}s`,
                    ip: req.ip
                });
            });
            next();
        });

        // 安全中間件
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    scriptSrc: ["'self'", "'unsafe-inline'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    imgSrc: ["'self'", "data:", "https:"],
                    connectSrc: ["'self'", "wss:", "https:"]
                }
            },
            hsts: {
                maxAge: 31536000,
                includeSubDomains: true,
                preload: true
            }
        }));

        // CORS
        this.app.use(cors({
            origin: (origin, callback) => {
                const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
                if (!origin || allowedOrigins.includes(origin)) {
                    callback(null, true);
                } else {
                    callback(new Error('Not allowed by CORS'));
                }
            },
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID']
        }));

        // 壓縮
        this.app.use(compression({
            level: 6,
            threshold: 1024,
            filter: (req, res) => {
                if (req.headers['x-no-compression']) {
                    return false;
                }
                return compression.filter(req, res);
            }
        }));

        // 解析請求體
        this.app.use(express.json({ 
            limit: '10mb',
            verify: (req, res, buf) => {
                req.rawBody = buf.toString('utf8');
            }
        }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

        // 信任代理
        this.app.set('trust proxy', true);

        // 全局速率限制
        this.app.use('/api/', this.security.apiLimiter);
    }

    setupRoutes() {
        // 健康檢查
        this.app.get('/health', async (req, res) => {
            try {
                await this.db.query('SELECT 1');
                await new Promise((resolve, reject) => {
                    this.db.redis.ping((err) => {
                        if (err) reject(err);
                        else resolve();
                    });
                });
                
                res.json({
                    status: 'healthy',
                    timestamp: new Date().toISOString(),
                    services: {
                        database: 'connected',
                        redis: 'connected',
                        kafka: this.kafka ? 'connected' : 'disconnected'
                    }
                });
            } catch (error) {
                res.status(503).json({
                    status: 'unhealthy',
                    error: error.message
                });
            }
        });

        // 就緒檢查
        this.app.get('/ready', (req, res) => {
            res.json({ ready: true });
        });

        // 指標端點
        this.app.get('/metrics', async (req, res) => {
            res.set('Content-Type', this.metrics.register.contentType);
            const metrics = await this.metrics.getMetrics();
            res.end(metrics);
        });

        // API 路由
        this.app.use('/api/auth', this.createAuthRouter());
        this.app.use('/api/users', this.createUserRouter());
        this.app.use('/api/bets', this.createBetRouter());
        this.app.use('/api/odds', this.createOddsRouter());
        this.app.use('/api/events', this.createEventRouter());
        this.app.use('/api/predictions', this.createPredictionRouter());
    }

    createAuthRouter() {
        const router = express.Router();

        // 註冊
        router.post('/register', this.security.authLimiter, async (req, res, next) => {
            try {
                const { username, email, password } = req.body;

                // 驗證輸入
                if (!username || !email || !password) {
                    return res.status(400).json({ error: 'Missing required fields' });
                }

                // 檢查用戶是否存在
                const existingUser = await this.db.query(
                    'SELECT id FROM users WHERE username = $1 OR email = $2',
                    [username, email]
                );

                if (existingUser.rows.length > 0) {
                    return res.status(409).json({ error: 'User already exists' });
                }

                // 創建用戶
                const hashedPassword = await this.security.hashPassword(password);
                const result = await this.db.query(
                    `INSERT INTO users (username, email, password_hash, balance) 
                     VALUES ($1, $2, $3, $4) 
                     RETURNING id, username, email, balance, created_at`,
                    [username, email, hashedPassword, 0]
                );

                const user = result.rows[0];
                const tokens = this.security.generateTokens(user);

                // 記錄註冊
                this.metrics.recordUserRegistration();
                logger.info('New user registered:', { userId: user.id, username });

                res.status(201).json({
                    user: {
                        id: user.id,
                        username: user.username,
                        email: user.email,
                        balance: user.balance
                    },
                    ...tokens
                });
            } catch (error) {
                next(error);
            }
        });

        // 登入
        router.post('/login', this.security.authLimiter, async (req, res, next) => {
            try {
                const { username, password } = req.body;

                if (!username || !password) {
                    return res.status(400).json({ error: 'Missing credentials' });
                }

                // 查找用戶
                const result = await this.db.query(
                    `SELECT id, username, email, password_hash, role, balance, is_active 
                     FROM users WHERE username = $1 OR email = $1`,
                    [username]
                );

                if (result.rows.length === 0) {
                    return res.status(401).json({ error: 'Invalid credentials' });
                }

                const user = result.rows[0];

                // 檢查帳戶狀態
                if (!user.is_active) {
                    return res.status(403).json({ error: 'Account is disabled' });
                }

                // 驗證密碼
                const isValid = await this.security.verifyPassword(password, user.password_hash);
                if (!isValid) {
                    return res.status(401).json({ error: 'Invalid credentials' });
                }

                // 生成令牌
                const tokens = this.security.generateTokens(user);

                // 更新最後登入時間
                await this.db.query(
                    'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = $1',
                    [user.id]
                );

                logger.info('User logged in:', { userId: user.id, username: user.username });

                res.json({
                    user: {
                        id: user.id,
                        username: user.username,
                        email: user.email,
                        balance: user.balance,
                        role: user.role
                    },
                    ...tokens
                });
            } catch (error) {
                next(error);
            }
        });

        // 刷新令牌
        router.post('/refresh', async (req, res, next) => {
            try {
                const { refreshToken } = req.body;

                if (!refreshToken) {
                    return res.status(400).json({ error: 'Refresh token required' });
                }

                // 驗證刷新令牌
                const decoded = this.security.verifyRefreshToken(refreshToken);

                // 獲取用戶資訊
                const result = await this.db.query(
                    'SELECT id, username, role FROM users WHERE id = $1 AND is_active = true',
                    [decoded.userId]
                );

                if (result.rows.length === 0) {
                    return res.status(401).json({ error: 'Invalid token' });
                }

                const user = result.rows[0];
                const tokens = this.security.generateTokens(user);

                res.json(tokens);
            } catch (error) {
                if (error.message === 'Invalid refresh token') {
                    return res.status(401).json({ error: error.message });
                }
                next(error);
            }
        });

        // 登出
        router.post('/logout', this.security.authenticateToken.bind(this.security), async (req, res) => {
            // 在實際應用中，您可能需要將令牌加入黑名單
            logger.info('User logged out:', { userId: req.user.userId });
            res.json({ message: 'Logged out successfully' });
        });

        return router;
    }

    createUserRouter() {
        const router = express.Router();

        // 獲取當前用戶資訊
        router.get('/me', this.security.authenticateToken.bind(this.security), async (req, res, next) => {
            try {
                const result = await this.db.query(
                    `SELECT id, username, email, balance, role, created_at, last_login 
                     FROM users WHERE id = $1`,
                    [req.user.userId]
                );

                if (result.rows.length === 0) {
                    return res.status(404).json({ error: 'User not found' });
                }

                res.json(result.rows[0]);
            } catch (error) {
                next(error);
            }
        });

        // 更新用戶資訊
        router.patch('/me', this.security.authenticateToken.bind(this.security), async (req, res, next) => {
            try {
                const { email } = req.body;
                const updates = [];
                const values = [];
                let paramCount = 1;

                if (email) {
                    updates.push(`email = $${paramCount++}`);
                    values.push(email);
                }

                if (updates.length === 0) {
                    return res.status(400).json({ error: 'No valid updates provided' });
                }

                values.push(req.user.userId);
                const query = `
                    UPDATE users 
                    SET ${updates.join(', ')}, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = $${paramCount}
                    RETURNING id, username, email, balance, role
                `;

                const result = await this.db.query(query, values);
                res.json(result.rows[0]);
            } catch (error) {
                next(error);
            }
        });

        // 獲取用戶交易歷史
        router.get('/me/transactions', this.security.authenticateToken.bind(this.security), async (req, res, next) => {
            try {
                const { limit = 50, offset = 0 } = req.query;

                const result = await this.db.query(
                    `SELECT 
                        t.transaction_id,
                        t.type,
                        t.amount,
                        t.description,
                        t.created_at,
                        t.status
                     FROM transactions t
                     WHERE t.user_id = $1
                     ORDER BY t.created_at DESC
                     LIMIT $2 OFFSET $3`,
                    [req.user.userId, limit, offset]
                );

                res.json({
                    transactions: result.rows,
                    pagination: {
                        limit: parseInt(limit),
                        offset: parseInt(offset)
                    }
                });
            } catch (error) {
                next(error);
            }
        });

        return router;
    }

    createBetRouter() {
        const router = express.Router();

        // 下注
        router.post('/', 
            this.security.authenticateToken.bind(this.security),
            this.security.betLimiter,
            async (req, res, next) => {
            try {
                const { eventId, marketType, selection, stake, odds } = req.body;

                // 驗證輸入
                if (!eventId || !marketType || !selection || !stake || !odds) {
                    return res.status(400).json({ error: 'Missing required fields' });
                }

                if (stake <= 0 || odds <= 1) {
                    return res.status(400).json({ error: 'Invalid stake or odds' });
                }

                // 使用事務處理下注
                const result = await this.db.transaction(async (client) => {
                    // 檢查用戶餘額
                    const userResult = await client.query(
                        'SELECT balance FROM users WHERE id = $1 FOR UPDATE',
                        [req.user.userId]
                    );

                    if (userResult.rows.length === 0) {
                        throw new Error('User not found');
                    }

                    const balance = parseFloat(userResult.rows[0].balance);
                    if (balance < stake) {
                        throw new Error('Insufficient balance');
                    }

                    // 檢查賽事狀態
                    const eventResult = await client.query(
                        'SELECT status, start_time FROM events WHERE event_id = $1',
                        [eventId]
                    );

                    if (eventResult.rows.length === 0) {
                        throw new Error('Event not found');
                    }

                    const event = eventResult.rows[0];
                    if (event.status !== 'scheduled' || new Date(event.start_time) < new Date()) {
                        throw new Error('Event is not available for betting');
                    }

                    // 創建下注記錄
                    const betResult = await client.query(
                        `INSERT INTO bet_transactions 
                         (user_id, event_id, market_type, selection, stake, odds, potential_payout, placed_at)
                         VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                         RETURNING transaction_id, stake, odds, potential_payout`,
                        [req.user.userId, eventId, marketType, selection, stake, odds, stake * odds]
                    );

                    // 更新用戶餘額
                    await client.query(
                        'UPDATE users SET balance = balance - $1 WHERE id = $2',
                        [stake, req.user.userId]
                    );

                    return betResult.rows[0];
                });

                // 記錄指標
                this.metrics.recordBet(marketType, selection, stake);

                logger.info('Bet placed:', {
                    userId: req.user.userId,
                    transactionId: result.transaction_id,
                    stake,
                    odds
                });

                res.status(201).json({
                    transactionId: result.transaction_id,
                    stake: result.stake,
                    odds: result.odds,
                    potentialPayout: result.potential_payout,
                    status: 'pending'
                });
            } catch (error) {
                if (error.message === 'Insufficient balance') {
                    return res.status(400).json({ error: error.message });
                }
                next(error);
            }
        });

        // 獲取用戶下注歷史
        router.get('/history', this.security.authenticateToken.bind(this.security), async (req, res, next) => {
            try {
                const { limit = 50, offset = 0, status } = req.query;

                let query = `
                    SELECT 
                        bt.transaction_id,
                        bt.event_id,
                        e.event_name,
                        bt.market_type,
                        bt.selection,
                        bt.stake,
                        bt.odds,
                        bt.potential_payout,
                        bt.bet_status,
                        bt.placed_at,
                        bt.settled_at,
                        bt.payout
                    FROM bet_transactions bt
                    JOIN events e ON bt.event_id = e.event_id
                    WHERE bt.user_id = $1
                `;

                const params = [req.user.userId];
                let paramIndex = 2;

                if (status) {
                    query += ` AND bt.bet_status = $${paramIndex++}`;
                    params.push(status);
                }

                query += ` ORDER BY bt.placed_at DESC LIMIT $${paramIndex++} OFFSET $${paramIndex}`;
                params.push(limit, offset);

                const result = await this.db.query(query, params);

                res.json({
                    bets: result.rows,
                    pagination: {
                        limit: parseInt(limit),
                        offset: parseInt(offset)
                    }
                });
            } catch (error) {
                next(error);
            }
        });

        return router;
    }

    createOddsRouter() {
        const router = express.Router();

        // 獲取賽事賠率
        router.get('/events/:eventId', async (req, res, next) => {
            try {
                const { eventId } = req.params;
                const cacheKey = `odds:${eventId}`;

                // 嘗試從緩存獲取
                const cached = await this.cache.get(cacheKey);
                if (cached) {
                    return res.json(cached);
                }

                // 從資料庫獲取
                const result = await this.db.query(
                    `SELECT 
                        o.bookmaker,
                        o.market_type,
                        o.odds_value,
                        o.timestamp,
                        o.is_available
                     FROM odds_history o
                     WHERE o.event_id = $1 
                       AND o.timestamp > NOW() - INTERVAL '5 minutes'
                       AND o.is_available = true
                     ORDER BY o.timestamp DESC`,
                    [eventId]
                );

                // 按博彩公司分組
                const oddsByBookmaker = {};
                result.rows.forEach(row => {
                    if (!oddsByBookmaker[row.bookmaker]) {
                        oddsByBookmaker[row.bookmaker] = {};
                    }
                    oddsByBookmaker[row.bookmaker][row.market_type] = {
                        odds: row.odds_value,
                        timestamp: row.timestamp
                    };
                });

                const response = {
                    eventId,
                    odds: oddsByBookmaker,
                    lastUpdate: new Date().toISOString()
                };

                // 緩存結果
                await this.cache.set(cacheKey, response, 30); // 30秒緩存

                res.json(response);
            } catch (error) {
                next(error);
            }
        });

        // 獲取實時賠率更新（SSE）
        router.get('/live/:eventId', async (req, res, next) => {
            const { eventId } = req.params;

            // 設置 SSE headers
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            });

            // 發送初始數據
            res.write(`data: ${JSON.stringify({ type: 'connected', eventId })}\n\n`);

            // 設置定期發送賠率更新
            const intervalId = setInterval(async () => {
                try {
                    const result = await this.db.query(
                        `SELECT 
                            bookmaker,
                            market_type,
                            odds_value,
                            timestamp
                         FROM odds_history
                         WHERE event_id = $1 
                           AND timestamp > NOW() - INTERVAL '10 seconds'
                         ORDER BY timestamp DESC
                         LIMIT 20`,
                        [eventId]
                    );

                    if (result.rows.length > 0) {
                        res.write(`data: ${JSON.stringify({
                            type: 'odds_update',
                            eventId,
                            odds: result.rows
                        })}\n\n`);
                    }
                } catch (error) {
                    logger.error('SSE error:', error);
                }
            }, 2000); // 每2秒更新

            // 清理連接
            req.on('close', () => {
                clearInterval(intervalId);
                res.end();
            });
        });

        return router;
    }

    createEventRouter() {
        const router = express.Router();

        // 獲取賽事列表
        router.get('/', async (req, res, next) => {
            try {
                const { sport, status = 'scheduled', limit = 50, offset = 0 } = req.query;

                let query = `
                    SELECT 
                        e.event_id,
                        e.sport_type,
                        e.event_name,
                        e.start_time,
                        e.status,
                        e.venue,
                        e.competition
                    FROM events e
                    WHERE e.status = $1
                `;

                const params = [status];
                let paramIndex = 2;

                if (sport) {
                    query += ` AND e.sport_type = $${paramIndex++}`;
                    params.push(sport);
                }

                query += ` AND e.start_time > NOW() ORDER BY e.start_time ASC LIMIT $${paramIndex++} OFFSET $${paramIndex}`;
                params.push(limit, offset);

                const result = await this.db.query(query, params);

                res.json({
                    events: result.rows,
                    pagination: {
                        limit: parseInt(limit),
                        offset: parseInt(offset)
                    }
                });
            } catch (error) {
                next(error);
            }
        });

        // 獲取單個賽事詳情
        router.get('/:eventId', async (req, res, next) => {
            try {
                const { eventId } = req.params;

                const result = await this.db.query(
                    `SELECT 
                        e.*,
                        COUNT(DISTINCT bt.transaction_id) as total_bets,
                        SUM(bt.stake) as total_stake
                     FROM events e
                     LEFT JOIN bet_transactions bt ON e.event_id = bt.event_id
                     WHERE e.event_id = $1
                     GROUP BY e.event_id`,
                    [eventId]
                );

                if (result.rows.length === 0) {
                    return res.status(404).json({ error: 'Event not found' });
                }

                res.json(result.rows[0]);
            } catch (error) {
                next(error);
            }
        });

        return router;
    }

    createPredictionRouter() {
        const router = express.Router();

        // 獲取賽馬預測
        router.get('/horse-racing/:raceId', async (req, res, next) => {
            try {
                const { raceId } = req.params;

                // 這裡應該調用機器學習模型
                // 為了示範，使用模擬數據
                const predictions = [
                    {
                        horseNumber: 3,
                        horseName: "Lucky Star",
                        winProbability: 0.285,
                        recommendedStake: 50,
                        expectedReturn: 175
                    },
                    {
                        horseNumber: 7,
                        horseName: "Thunder Bolt",
                        winProbability: 0.223,
                        recommendedStake: 40,
                        expectedReturn: 150
                    }
                ];

                res.json({
                    raceId,
                    predictions,
                    modelConfidence: 0.82,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // 獲取足球預測
        router.get('/football/:matchId', async (req, res, next) => {
            try {
                const { matchId } = req.params;

                // 模擬貝葉斯網絡預測結果
                const prediction = {
                    matchId,
                    homeWin: 0.45,
                    draw: 0.30,
                    awayWin: 0.25,
                    confidence: 0.78,
                    factors: {
                        homeForm: 'good',
                        awayForm: 'average',
                        homeAdvantage: 0.15,
                        headToHead: 'balanced'
                    },
                    recommendedBet: {
                        outcome: 'home',
                        stake: 100,
                        expectedValue: 1.12
                    }
                };

                res.json(prediction);
            } catch (error) {
                next(error);
            }
        });

        return router;
    }

    async initializeKafka() {
        try {
            this.kafka = new Kafka({
                clientId: 'betting-system',
                brokers: process.env.KAFKA_BROKERS?.split(',') || ['localhost:9092'],
                connectionTimeout: 30000,
                authenticationTimeout: 10000,
                retry: {
                    initialRetryTime: 100,
                    retries: 8
                }
            });

            const admin = this.kafka.admin();
            await admin.connect();

            // 創建必要的主題
            const topics = [
                { topic: 'odds-updates', numPartitions: 10 },
                { topic: 'bet-placements', numPartitions: 5 },
                { topic: 'arbitrage-opportunities', numPartitions: 3 }
            ];

            const existingTopics = await admin.listTopics();
            const topicsToCreate = topics.filter(t => !existingTopics.includes(t.topic));

            if (topicsToCreate.length > 0) {
                await admin.createTopics({
                    topics: topicsToCreate,
                    waitForLeaders: true
                });
            }

            await admin.disconnect();
            logger.info('Kafka initialized successfully');
        } catch (error) {
            logger.error('Kafka initialization failed:', error);
            // 不要讓 Kafka 失敗阻止應用啟動
        }
    }

    initializeWebSocket() {
        this.wsServer = new WebSocket.Server({
            port: process.env.WS_PORT || 8080,
            perMessageDeflate: false,
            clientTracking: true,
            maxPayload: 64 * 1024 // 64KB
        });

        this.wsServer.on('connection', (ws, req) => {
            const clientId = req.headers['x-client-id'] || `client-${Date.now()}`;
            logger.info('WebSocket client connected:', { clientId, ip: req.socket.remoteAddress });

            this.metrics.updateActiveConnections('websocket', 1);

            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    logger.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({ error: 'Invalid message format' }));
                }
            });

            ws.on('close', () => {
                logger.info('WebSocket client disconnected:', { clientId });
                this.metrics.updateActiveConnections('websocket', -1);
            });

            ws.on('error', (error) => {
                logger.error('WebSocket error:', { clientId, error: error.message });
            });

            // 發送歡迎消息
            ws.send(JSON.stringify({
                type: 'welcome',
                clientId,
                timestamp: new Date().toISOString()
            }));
        });

        // 心跳檢測
        const heartbeat = setInterval(() => {
            this.wsServer.clients.forEach((ws) => {
                if (ws.isAlive === false) {
                    return ws.terminate();
                }
                ws.isAlive = false;
                ws.ping();
            });
        }, 30000);

        this.wsServer.on('close', () => {
            clearInterval(heartbeat);
        });
    }

    async handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'subscribe':
                // 訂閱特定賽事的賠率更新
                if (data.eventId) {
                    ws.eventSubscriptions = ws.eventSubscriptions || new Set();
                    ws.eventSubscriptions.add(data.eventId);
                    ws.send(JSON.stringify({
                        type: 'subscribed',
                        eventId: data.eventId
                    }));
                }
                break;

            case 'unsubscribe':
                if (data.eventId && ws.eventSubscriptions) {
                    ws.eventSubscriptions.delete(data.eventId);
                    ws.send(JSON.stringify({
                        type: 'unsubscribed',
                        eventId: data.eventId
                    }));
                }
                break;

            case 'ping':
                ws.send(JSON.stringify({ type: 'pong' }));
                break;

            default:
                ws.send(JSON.stringify({ error: 'Unknown message type' }));
        }
    }

    setupErrorHandling() {
        // 404 處理
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Not Found',
                path: req.path
            });
        });

        // 全局錯誤處理
        this.app.use((err, req, res, next) => {
            logger.error('Unhandled error:', {
                error: err.message,
                stack: err.stack,
                path: req.path,
                method: req.method
            });

            // 資料庫錯誤
            if (err.code === '23505') {
                return res.status(409).json({ error: 'Duplicate entry' });
            }

            if (err.code === '23503') {
                return res.status(400).json({ error: 'Foreign key violation' });
            }

            // 驗證錯誤
            if (err.name === 'ValidationError') {
                return res.status(400).json({ error: err.message });
            }

            // 默認錯誤響應
            res.status(err.status || 500).json({
                error: err.message || 'Internal Server Error',
                ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
            });
        });
    }

    async start(port = process.env.PORT || 3000) {
        const server = this.app.listen(port, () => {
            logger.info(`Server running on port ${port}`);
            logger.info(`WebSocket server running on port ${process.env.WS_PORT || 8080}`);
        });

        // 優雅關機
        process.on('SIGTERM', async () => {
            logger.info('SIGTERM signal received: closing HTTP server');
            
            server.close(() => {
                logger.info('HTTP server closed');
            });

            // 關閉 WebSocket 服務器
            if (this.wsServer) {
                this.wsServer.close(() => {
                    logger.info('WebSocket server closed');
                });
            }

            // 關閉資料庫連接
            try {
                await this.db.pool.end();
                this.db.redis.quit();
                logger.info('Database connections closed');
            } catch (error) {
                logger.error('Error closing database connections:', error);
            }

            process.exit(0);
        });

        return server;
    }
}

// 啟動應用程式
async function main() {
    try {
        const app = new BettingSystemApp();
        await app.initialize();
        await app.start();
    } catch (error) {
        logger.error('Failed to start application:', error);
        process.exit(1);
    }
}

// 處理未捕獲的異常
process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// 如果直接運行此文件，啟動服務器
if (require.main === module) {
    main();
}

module.exports = BettingSystemApp;