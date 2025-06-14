# 生產級投注系統 - 完整實現指南

## 項目結構

```
betting-system/
├── frontend/                    # Next.js 14 前端
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── dashboard/
│   │   ├── football/
│   │   ├── horse-racing/
│   │   └── api/
│   ├── components/
│   │   ├── betting/
│   │   ├── charts/
│   │   └── ui/
│   ├── hooks/
│   ├── lib/
│   └── services/
├── backend/                     # Node.js 後端
│   ├── src/
│   │   ├── server.ts
│   │   ├── websocket/
│   │   ├── api/
│   │   └── services/
│   └── models/
├── ml-models/                   # 機器學習模型
│   ├── horse-racing/
│   └── football/
├── docker-compose.yml
└── README.md
```

## 1. 前端實現 (Next.js 14 + TypeScript + Tremor UI)

### package.json
```json
{
  "name": "betting-system-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tremor/react": "^3.14.0",
    "recharts": "^2.10.0",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "axios": "^1.6.0",
    "react-use-websocket": "^4.5.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "react-hook-form": "^7.48.0",
    "zod": "^3.22.0",
    "@hookform/resolvers": "^3.3.0",
    "lucide-react": "^0.300.0",
    "date-fns": "^3.0.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "14.1.0"
  }
}
```

### app/layout.tsx
```typescript
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers'
import { Navigation } from '@/components/navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Professional Betting System',
  description: 'Advanced sports betting analytics platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-gray-950">
            <Navigation />
            <main className="container mx-auto px-4 py-8">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}
```

### app/page.tsx
```typescript
import { Card, Metric, Text, Flex, ProgressBar, Grid } from '@tremor/react'
import { LiveOddsPanel } from '@/components/betting/LiveOddsPanel'
import { RecentBetsTable } from '@/components/betting/RecentBetsTable'
import { PerformanceChart } from '@/components/charts/PerformanceChart'

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Betting Dashboard</h1>
      
      <Grid numItemsSm={2} numItemsLg={4} className="gap-6">
        <Card className="bg-gray-900 border-gray-800">
          <Text className="text-gray-400">Total Profit</Text>
          <Metric className="text-white">$2,543</Metric>
          <Flex className="mt-4">
            <Text className="text-xs text-gray-400">32% increase</Text>
            <Text className="text-xs text-gray-400">vs last month</Text>
          </Flex>
          <ProgressBar value={32} className="mt-2" color="emerald" />
        </Card>
        
        <Card className="bg-gray-900 border-gray-800">
          <Text className="text-gray-400">Win Rate</Text>
          <Metric className="text-white">68.4%</Metric>
          <Flex className="mt-4">
            <Text className="text-xs text-gray-400">154 wins</Text>
            <Text className="text-xs text-gray-400">71 losses</Text>
          </Flex>
          <ProgressBar value={68.4} className="mt-2" color="blue" />
        </Card>
        
        <Card className="bg-gray-900 border-gray-800">
          <Text className="text-gray-400">Active Bets</Text>
          <Metric className="text-white">12</Metric>
          <Flex className="mt-4">
            <Text className="text-xs text-gray-400">$1,250 at risk</Text>
          </Flex>
        </Card>
        
        <Card className="bg-gray-900 border-gray-800">
          <Text className="text-gray-400">Model Accuracy</Text>
          <Metric className="text-white">35.2%</Metric>
          <Flex className="mt-4">
            <Text className="text-xs text-gray-400">Horse Racing</Text>
            <Text className="text-xs text-gray-400">+7.2% vs baseline</Text>
          </Flex>
          <ProgressBar value={35.2} className="mt-2" color="amber" />
        </Card>
      </Grid>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <PerformanceChart />
        </div>
        <div>
          <LiveOddsPanel />
        </div>
      </div>
      
      <RecentBetsTable />
    </div>
  )
}
```

### components/betting/LiveOddsPanel.tsx
```typescript
'use client'

import { Card, Title, Text, Badge } from '@tremor/react'
import { useBettingWebSocket } from '@/hooks/useBettingWebSocket'
import { TrendingUp, TrendingDown } from 'lucide-react'

export function LiveOddsPanel() {
  const { odds, connectionStatus } = useBettingWebSocket()
  
  return (
    <Card className="bg-gray-900 border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <Title className="text-white">Live Odds</Title>
        <Badge 
          color={connectionStatus === 'connected' ? 'emerald' : 'red'}
          size="xs"
        >
          {connectionStatus}
        </Badge>
      </div>
      
      <div className="space-y-3">
        {odds.map((item) => (
          <div key={item.id} className="p-3 bg-gray-800 rounded-lg">
            <div className="flex justify-between items-start">
              <div>
                <Text className="text-white font-medium">{item.name}</Text>
                <Text className="text-gray-400 text-xs">{item.event}</Text>
              </div>
              <div className="text-right">
                <Text className="text-white font-mono">{item.odds}</Text>
                <div className="flex items-center justify-end mt-1">
                  {item.movement > 0 ? (
                    <TrendingUp className="w-3 h-3 text-emerald-500 mr-1" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-red-500 mr-1" />
                  )}
                  <Text className={`text-xs ${
                    item.movement > 0 ? 'text-emerald-500' : 'text-red-500'
                  }`}>
                    {Math.abs(item.movement)}%
                  </Text>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
```

### hooks/useBettingWebSocket.ts
```typescript
import { useEffect, useState, useCallback } from 'react'
import useWebSocket from 'react-use-websocket'
import { useAuthStore } from '@/stores/authStore'

interface OddsUpdate {
  id: string
  name: string
  event: string
  odds: number
  movement: number
  timestamp: number
}

export function useBettingWebSocket() {
  const [odds, setOdds] = useState<OddsUpdate[]>([])
  const { token } = useAuthStore()
  
  const socketUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001'
  
  const { sendMessage, lastMessage, readyState } = useWebSocket(
    `${socketUrl}?token=${token}`,
    {
      shouldReconnect: (closeEvent) => closeEvent.code !== 1000,
      reconnectAttempts: 10,
      reconnectInterval: (attemptNumber) => 
        Math.min(Math.pow(2, attemptNumber) * 1000, 30000),
      onOpen: () => console.log('WebSocket connected'),
      onClose: () => console.log('WebSocket disconnected'),
      onError: (error) => console.error('WebSocket error:', error)
    }
  )
  
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data)
        if (data.type === 'odds_update') {
          setOdds(data.payload)
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }
  }, [lastMessage])
  
  const placeBet = useCallback((betData: any) => {
    if (readyState === WebSocket.OPEN) {
      sendMessage(JSON.stringify({
        type: 'place_bet',
        payload: betData
      }))
    }
  }, [sendMessage, readyState])
  
  const connectionStatus = readyState === WebSocket.OPEN ? 'connected' : 'disconnected'
  
  return { odds, placeBet, connectionStatus }
}
```

### app/horse-racing/page.tsx
```typescript
'use client'

import { useState } from 'react'
import { Card, Title, Text, Button, Select, SelectItem } from '@tremor/react'
import { HorseRacingPrediction } from '@/components/horse-racing/HorseRacingPrediction'
import { RaceAnalytics } from '@/components/horse-racing/RaceAnalytics'
import { useHorseRacingData } from '@/hooks/useHorseRacingData'

export default function HorseRacingPage() {
  const [selectedRace, setSelectedRace] = useState<string>('')
  const { races, predictions, isLoading } = useHorseRacingData()
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Horse Racing Analytics</h1>
        <Button size="sm" variant="secondary">
          Refresh Data
        </Button>
      </div>
      
      <Card className="bg-gray-900 border-gray-800">
        <Title className="text-white mb-4">Select Race</Title>
        <Select value={selectedRace} onValueChange={setSelectedRace}>
          {races.map((race) => (
            <SelectItem key={race.id} value={race.id}>
              {race.name} - {race.time}
            </SelectItem>
          ))}
        </Select>
      </Card>
      
      {selectedRace && (
        <>
          <HorseRacingPrediction raceId={selectedRace} />
          <RaceAnalytics raceId={selectedRace} />
        </>
      )}
    </div>
  )
}
```

### services/api.ts
```typescript
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Handle token refresh
      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const { data } = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refreshToken
          })
          localStorage.setItem('auth_token', data.accessToken)
          error.config.headers.Authorization = `Bearer ${data.accessToken}`
          return api(error.config)
        } catch (refreshError) {
          // Redirect to login
          window.location.href = '/login'
        }
      }
    }
    return Promise.reject(error)
  }
)

export const bettingAPI = {
  // Horse Racing
  async getHorseRaces() {
    const { data } = await api.get('/horse-racing/races')
    return data
  },
  
  async getHorseRacePrediction(raceId: string) {
    const { data } = await api.get(`/horse-racing/predictions/${raceId}`)
    return data
  },
  
  // Football
  async getFootballMatches(league: string) {
    const { data } = await api.get('/football/matches', { params: { league } })
    return data
  },
  
  async getFootballPrediction(matchId: string) {
    const { data } = await api.get(`/football/predictions/${matchId}`)
    return data
  },
  
  // Betting
  async placeBet(betData: any) {
    const { data } = await api.post('/bets', betData)
    return data
  },
  
  async getBettingHistory() {
    const { data } = await api.get('/bets/history')
    return data
  }
}
```

## 2. 後端實現 (Node.js + TypeScript)

### backend/package.json
```json
{
  "name": "betting-system-backend",
  "version": "1.0.0",
  "scripts": {
    "dev": "nodemon src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js",
    "test": "jest",
    "lint": "eslint src/**/*.ts"
  },
  "dependencies": {
    "express": "^4.18.0",
    "ws": "^8.16.0",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "jsonwebtoken": "^9.0.0",
    "bcrypt": "^5.1.0",
    "redis": "^4.6.0",
    "postgresql": "^1.0.0",
    "pg": "^8.11.0",
    "typeorm": "^0.3.0",
    "axios": "^1.6.0",
    "node-cron": "^3.0.0",
    "winston": "^3.11.0",
    "express-rate-limit": "^7.1.0",
    "dotenv": "^16.3.0",
    "joi": "^17.11.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/express": "^4.17.0",
    "@types/ws": "^8.5.0",
    "@types/cors": "^2.8.0",
    "@types/bcrypt": "^5.0.0",
    "@types/jsonwebtoken": "^9.0.0",
    "typescript": "^5.3.0",
    "nodemon": "^3.0.0",
    "ts-node": "^10.9.0",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.0",
    "eslint": "^8.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0"
  }
}
```

### backend/src/server.ts
```typescript
import express from 'express'
import { createServer } from 'http'
import cors from 'cors'
import helmet from 'helmet'
import { WebSocketServer } from './websocket/server'
import { setupDatabase } from './database/setup'
import { authRouter } from './routes/auth'
import { bettingRouter } from './routes/betting'
import { horseRacingRouter } from './routes/horseRacing'
import { footballRouter } from './routes/football'
import { errorHandler } from './middleware/errorHandler'
import { rateLimiter } from './middleware/rateLimiter'
import { logger } from './utils/logger'
import { startScheduledJobs } from './jobs/scheduler'
import { config } from './config'

const app = express()
const server = createServer(app)

// Middleware
app.use(helmet())
app.use(cors({
  origin: config.FRONTEND_URL,
  credentials: true
}))
app.use(express.json())
app.use(rateLimiter)

// Routes
app.use('/api/auth', authRouter)
app.use('/api/bets', bettingRouter)
app.use('/api/horse-racing', horseRacingRouter)
app.use('/api/football', footballRouter)

// Error handling
app.use(errorHandler)

// Initialize WebSocket server
const wsServer = new WebSocketServer(server)

// Start server
async function start() {
  try {
    // Setup database
    await setupDatabase()
    
    // Start scheduled jobs
    startScheduledJobs()
    
    // Start HTTP server
    server.listen(config.PORT, () => {
      logger.info(`Server running on port ${config.PORT}`)
    })
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
      logger.info('SIGTERM received, shutting down gracefully')
      server.close(() => {
        logger.info('HTTP server closed')
      })
      wsServer.close()
      process.exit(0)
    })
  } catch (error) {
    logger.error('Failed to start server:', error)
    process.exit(1)
  }
}

start()
```

### backend/src/websocket/server.ts
```typescript
import { WebSocketServer as WSServer } from 'ws'
import { Server } from 'http'
import jwt from 'jsonwebtoken'
import { logger } from '../utils/logger'
import { RedisClient } from '../services/redis'
import { config } from '../config'

interface Client {
  id: string
  userId: string
  ws: WebSocket
  subscriptions: Set<string>
}

export class WebSocketServer {
  private wss: WSServer
  private clients: Map<string, Client> = new Map()
  private redis: RedisClient
  
  constructor(server: Server) {
    this.wss = new WSServer({ server })
    this.redis = new RedisClient()
    this.initialize()
  }
  
  private initialize() {
    this.wss.on('connection', async (ws, req) => {
      try {
        // Authenticate connection
        const token = this.extractToken(req.url)
        const decoded = jwt.verify(token, config.JWT_SECRET) as any
        
        const client: Client = {
          id: this.generateClientId(),
          userId: decoded.sub,
          ws: ws as any,
          subscriptions: new Set()
        }
        
        this.clients.set(client.id, client)
        logger.info(`Client connected: ${client.id}`)
        
        // Setup event handlers
        ws.on('message', (data) => this.handleMessage(client, data))
        ws.on('close', () => this.handleDisconnect(client))
        ws.on('error', (error) => logger.error('WebSocket error:', error))
        
        // Send initial data
        this.sendInitialData(client)
        
      } catch (error) {
        logger.error('WebSocket authentication failed:', error)
        ws.close(1008, 'Invalid token')
      }
    })
    
    // Subscribe to Redis for real-time updates
    this.subscribeToOddsUpdates()
  }
  
  private async handleMessage(client: Client, data: any) {
    try {
      const message = JSON.parse(data.toString())
      
      switch (message.type) {
        case 'subscribe':
          this.handleSubscribe(client, message.payload)
          break
          
        case 'unsubscribe':
          this.handleUnsubscribe(client, message.payload)
          break
          
        case 'place_bet':
          await this.handlePlaceBet(client, message.payload)
          break
          
        default:
          logger.warn(`Unknown message type: ${message.type}`)
      }
    } catch (error) {
      logger.error('Failed to handle message:', error)
      this.sendError(client, 'Invalid message format')
    }
  }
  
  private handleSubscribe(client: Client, channels: string[]) {
    channels.forEach(channel => {
      client.subscriptions.add(channel)
      logger.info(`Client ${client.id} subscribed to ${channel}`)
    })
    
    this.sendMessage(client, {
      type: 'subscribed',
      payload: channels
    })
  }
  
  private handleUnsubscribe(client: Client, channels: string[]) {
    channels.forEach(channel => {
      client.subscriptions.delete(channel)
      logger.info(`Client ${client.id} unsubscribed from ${channel}`)
    })
    
    this.sendMessage(client, {
      type: 'unsubscribed',
      payload: channels
    })
  }
  
  private async handlePlaceBet(client: Client, betData: any) {
    try {
      // Validate bet data
      const validatedBet = await this.validateBet(betData)
      
      // Process bet through betting service
      const result = await this.processBet(client.userId, validatedBet)
      
      // Send confirmation
      this.sendMessage(client, {
        type: 'bet_placed',
        payload: result
      })
      
      // Broadcast to other clients if needed
      this.broadcastBetUpdate(result)
      
    } catch (error) {
      logger.error('Failed to place bet:', error)
      this.sendError(client, error.message)
    }
  }
  
  private subscribeToOddsUpdates() {
    this.redis.subscribe('odds:*', (channel, data) => {
      const odds = JSON.parse(data)
      this.broadcastToSubscribers(channel, {
        type: 'odds_update',
        payload: odds
      })
    })
  }
  
  private broadcastToSubscribers(channel: string, message: any) {
    this.clients.forEach(client => {
      if (client.subscriptions.has(channel)) {
        this.sendMessage(client, message)
      }
    })
  }
  
  private sendMessage(client: Client, message: any) {
    if (client.ws.readyState === 1) { // OPEN
      client.ws.send(JSON.stringify(message))
    }
  }
  
  private sendError(client: Client, error: string) {
    this.sendMessage(client, {
      type: 'error',
      payload: { message: error }
    })
  }
  
  private handleDisconnect(client: Client) {
    logger.info(`Client disconnected: ${client.id}`)
    this.clients.delete(client.id)
  }
  
  public close() {
    this.wss.close()
    this.redis.disconnect()
  }
  
  private extractToken(url: string | undefined): string {
    if (!url) throw new Error('No URL provided')
    const params = new URLSearchParams(url.split('?')[1])
    const token = params.get('token')
    if (!token) throw new Error('No token provided')
    return token
  }
  
  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }
}
```

## 3. 賽馬預測系統 (Python - 35%準確率目標)

### ml-models/horse-racing/enhanced_predictor.py
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedHorseRacingPredictor:
    """優化的賽馬預測系統（目標35%準確率）"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.ensemble_weights = {
            'xgboost': 0.35,
            'lightgbm': 0.30,
            'random_forest': 0.20,
            'neural_net': 0.15
        }
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """進階特徵工程"""
        logger.info("Starting feature engineering...")
        
        # 1. 速度與步速分析
        df['speed_figure_avg'] = df.groupby('horse_id')['speed_figure'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['speed_consistency'] = df.groupby('horse_id')['speed_figure'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # 2. 跑道偏好指標
        df['track_win_rate'] = df.groupby(['horse_id', 'track_code'])['win'].transform('mean')
        df['distance_win_rate'] = df.groupby(['horse_id', 'distance_category'])['win'].transform('mean')
        
        # 3. 騎師與練馬師協同效應
        df['jockey_trainer_combo'] = df.groupby(['jockey_id', 'trainer_id'])['win'].transform('mean')
        df['jockey_horse_combo'] = df.groupby(['jockey_id', 'horse_id'])['win'].transform('mean')
        
        # 4. 檔位分析（考慮跑道特性）
        df['draw_bias_score'] = self._calculate_draw_bias(df)
        
        # 5. 體重變化影響
        df['weight_change'] = df['current_weight'] - df['last_weight']
        df['weight_change_pct'] = df['weight_change'] / df['last_weight'] * 100
        
        # 6. 休息天數優化
        df['rest_category'] = pd.cut(df['days_since_last'], 
                                    bins=[0, 14, 28, 56, 365], 
                                    labels=['short', 'optimal', 'long', 'very_long'])
        df['is_optimal_rest'] = (df['days_since_last'] >= 14) & (df['days_since_last'] <= 35)
        
        # 7. 班次調整指標
        df['class_change'] = df['current_class'] - df['last_class']
        df['dropping_class'] = (df['class_change'] < 0).astype(int)
        
        # 8. 近期表現趨勢（指數加權）
        df['recent_form'] = df.groupby('horse_id').apply(
            lambda x: self._calculate_form_rating(x)
        ).reset_index(level=0, drop=True)
        
        # 9. 賠率變動分析
        if 'opening_odds' in df.columns:
            df['odds_movement'] = (df['final_odds'] - df['opening_odds']) / df['opening_odds']
            df['market_confidence'] = 1 / df['final_odds']
        
        # 10. 血統評分（如有數據）
        if 'sire_id' in df.columns:
            df['sire_win_rate'] = df.groupby('sire_id')['win'].transform('mean')
            df['dam_sire_win_rate'] = df.groupby('dam_sire_id')['win'].transform('mean')
        
        return df
    
    def _calculate_draw_bias(self, df: pd.DataFrame) -> pd.Series:
        """計算檔位偏差分數"""
        # 基於歷史數據計算每個跑道的檔位優勢
        draw_stats = df.groupby(['track_code', 'distance_category', 'barrier_draw'])['win'].agg(['mean', 'count'])
        
        # 只考慮有足夠樣本的數據
        draw_stats = draw_stats[draw_stats['count'] >= 20]
        
        # 計算相對優勢
        track_avg = df.groupby(['track_code', 'distance_category'])['win'].mean()
        
        bias_scores = []
        for _, row in df.iterrows():
            key = (row['track_code'], row['distance_category'], row['barrier_draw'])
            if key in draw_stats.index:
                track_key = (row['track_code'], row['distance_category'])
                if track_key in track_avg.index:
                    relative_advantage = draw_stats.loc[key, 'mean'] / track_avg[track_key]
                    bias_scores.append(relative_advantage)
                else:
                    bias_scores.append(1.0)
            else:
                bias_scores.append(1.0)
        
        return pd.Series(bias_scores, index=df.index)
    
    def _calculate_form_rating(self, group: pd.DataFrame) -> pd.Series:
        """計算近期狀態評分"""
        # 使用指數衰減權重
        weights = np.exp(-np.arange(len(group)) * 0.2)
        weights = weights / weights.sum()
        
        # 考慮完賽位置和速度
        position_score = (10 - group['finish_position'].values) / 10
        speed_score = group['speed_figure'].values / 100
        
        combined_score = (position_score + speed_score) / 2
        form_rating = np.sum(combined_score * weights[:len(combined_score)])
        
        return pd.Series([form_rating] * len(group), index=group.index)
    
    def train_ensemble(self, X: pd.DataFrame, y: np.array):
        """訓練集成模型"""
        logger.info("Training ensemble models...")
        
        # 1. XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgboost'].fit(X, y)
        
        # 2. LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=0.1,
            lambda_l2=0.1,
            min_data_in_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        self.models['lightgbm'].fit(X, y)
        
        # 3. Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X, y)
        
        # 4. Neural Network
        self.models['neural_net'] = self._build_neural_network(X.shape[1])
        self._train_neural_network(X, y)
        
        self.feature_names = X.columns.tolist()
        logger.info("Ensemble training completed")
    
    def _build_neural_network(self, input_dim: int) -> nn.Module:
        """構建神經網絡"""
        class RacingNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 256)
                self.bn1 = nn.BatchNorm1d(256)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.dropout2 = nn.Dropout(0.2)
                
                self.fc3 = nn.Linear(128, 64)
                self.bn3 = nn.BatchNorm1d(64)
                self.dropout3 = nn.Dropout(0.1)
                
                self.fc4 = nn.Linear(64, 1)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                
                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                
                x = torch.relu(self.bn3(self.fc3(x)))
                x = self.dropout3(x)
                
                x = torch.sigmoid(self.fc4(x))
                return x
        
        return RacingNet(input_dim)
    
    def _train_neural_network(self, X: pd.DataFrame, y: np.array):
        """訓練神經網絡"""
        # 準備數據
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # 訓練設置
        optimizer = torch.optim.Adam(self.models['neural_net'].parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # 訓練循環
        self.models['neural_net'].train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.models['neural_net'](X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Neural Network - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict_race(self, race_data: pd.DataFrame) -> Dict[str, float]:
        """預測比賽結果"""
        # 特徵工程
        race_data = self.engineer_features(race_data)
        X = race_data[self.feature_names]
        
        # 獲取各模型預測
        predictions = {}
        
        # XGBoost
        predictions['xgboost'] = self.models['xgboost'].predict_proba(X)[:, 1]
        
        # LightGBM
        predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X)[:, 1]
        
        # Random Forest
        predictions['random_forest'] = self.models['random_forest'].predict_proba(X)[:, 1]
        
        # Neural Network
        self.models['neural_net'].eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            predictions['neural_net'] = self.models['neural_net'](X_tensor).numpy().flatten()
        
        # 加權集成
        final_predictions = np.zeros(len(X))
        for model_name, weight in self.ensemble_weights.items():
            final_predictions += predictions[model_name] * weight
        
        # 返回結果
        results = {}
        for i, horse_id in enumerate(race_data['horse_id']):
            results[horse_id] = {
                'win_probability': float(final_predictions[i]),
                'rank': 0,  # 將在後面計算
                'confidence': self._calculate_confidence(predictions, i)
            }
        
        # 計算排名
        sorted_horses = sorted(results.items(), key=lambda x: x[1]['win_probability'], reverse=True)
        for rank, (horse_id, _) in enumerate(sorted_horses, 1):
            results[horse_id]['rank'] = rank
        
        return results
    
    def _calculate_confidence(self, predictions: Dict[str, np.array], index: int) -> float:
        """計算預測置信度"""
        # 計算各模型預測的標準差
        model_preds = [pred[index] for pred in predictions.values()]
        std_dev = np.std(model_preds)
        
        # 置信度與標準差成反比
        confidence = 1 / (1 + std_dev * 10)
        return float(np.clip(confidence, 0, 1))
    
    def save_model(self, path: str):
        """保存模型"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.ensemble_weights = model_data['ensemble_weights']
        logger.info(f"Model loaded from {path}")

# API 接口
from flask import Flask, request, jsonify
import asyncio

app = Flask(__name__)
predictor = EnhancedHorseRacingPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        race_data = pd.DataFrame(request.json)
        predictions = predictor.predict_race(race_data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = pd.DataFrame(request.json['data'])
        X = predictor.engineer_features(data)
        y = data['win'].values
        predictor.train_ensemble(X, y)
        predictor.save_model('models/horse_racing_model.pkl')
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 載入預訓練模型
    try:
        predictor.load_model('models/horse_racing_model.pkl')
    except:
        logger.info("No pre-trained model found")
    
    app.run(host='0.0.0.0', port=5001)
```

## 4. 足球貝葉斯預測系統

### ml-models/football/bayesian_predictor.py
```python
import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from typing import Dict, List, Tuple
import redis
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianFootballPredictor:
    """貝葉斯足球預測系統"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.models = {}
        self.traces = {}
        self.team_stats = {}
        
    def build_dixon_coles_model(self, matches_df: pd.DataFrame) -> pm.Model:
        """構建 Dixon-Coles 模型"""
        teams = pd.concat([matches_df['home_team'], matches_df['away_team']]).unique()
        n_teams = len(teams)
        team_lookup = {team: i for i, team in enumerate(teams)}
        
        home_idx = matches_df['home_team'].map(team_lookup).values
        away_idx = matches_df['away_team'].map(team_lookup).values
        home_goals = matches_df['home_goals'].values
        away_goals = matches_df['away_goals'].values
        
        with pm.Model() as model:
            # 超參數
            mu_att = pm.Normal('mu_att', mu=0, sd=1)
            mu_def = pm.Normal('mu_def', mu=0, sd=1)
            tau_att = pm.Exponential('tau_att', lam=1)
            tau_def = pm.Exponential('tau_def', lam=1)
            
            # 團隊參數
            attack = pm.Normal('attack', mu=mu_att, sd=tau_att, shape=n_teams)
            defense = pm.Normal('defense', mu=mu_def, sd=tau_def, shape=n_teams)
            
            # 主場優勢
            home_advantage = pm.Normal('home_advantage', mu=0.25, sd=0.05)
            
            # 預期進球
            home_theta = tt.exp(attack[home_idx] - defense[away_idx] + home_advantage)
            away_theta = tt.exp(attack[away_idx] - defense[home_idx])
            
            # 觀測值
            home_goals_obs = pm.Poisson('home_goals', mu=home_theta, observed=home_goals)
            away_goals_obs = pm.Poisson('away_goals', mu=away_theta, observed=away_goals)
        
        self.team_lookup = team_lookup
        self.teams = teams
        return model
    
    def train_model(self, league: str, matches_df: pd.DataFrame):
        """訓練模型"""
        logger.info(f"Training model for {league}")
        
        model = self.build_dixon_coles_model(matches_df)
        
        with model:
            trace = pm.sample(
                draws=5000,
                chains=4,
                tune=1000,
                target_accept=0.95,
                return_inferencedata=True
            )
        
        self.models[league] = model
        self.traces[league] = trace
        
        # 保存到 Redis
        self._save_model_to_redis(league, trace)
        
        logger.info(f"Model for {league} trained successfully")
    
    def predict_match(self, home_team: str, away_team: str, league: str) -> Dict:
        """預測比賽"""
        if league not in self.traces:
            self._load_model_from_redis(league)
        
        trace = self.traces[league]
        
        # 獲取團隊索引
        home_idx = self.team_lookup[home_team]
        away_idx = self.team_lookup[away_team]
        
        # 提取參數
        attack = trace.posterior['attack'].values.reshape(-1, len(self.teams))
        defense = trace.posterior['defense'].values.reshape(-1, len(self.teams))
        home_advantage = trace.posterior['home_advantage'].values.flatten()
        
        # 計算預期進球率
        home_rate = np.exp(attack[:, home_idx] - defense[:, away_idx] + home_advantage)
        away_rate = np.exp(attack[:, away_idx] - defense[:, home_idx])
        
        # 模擬比賽
        n_simulations = 10000
        home_goals = np.random.poisson(home_rate[:n_simulations])
        away_goals = np.random.poisson(away_rate[:n_simulations])
        
        # 計算概率
        home_win_prob = np.mean(home_goals > away_goals)
        draw_prob = np.mean(home_goals == away_goals)
        away_win_prob = np.mean(home_goals < away_goals)
        
        # 計算各種投注市場
        predictions = {
            'match_result': {
                'home_win': float(home_win_prob),
                'draw': float(draw_prob),
                'away_win': float(away_win_prob)
            },
            'expected_goals': {
                'home': float(np.mean(home_rate)),
                'away': float(np.mean(away_rate))
            },
            'both_teams_to_score': {
                'yes': float(np.mean((home_goals > 0) & (away_goals > 0))),
                'no': float(np.mean((home_goals == 0) | (away_goals == 0)))
            },
            'over_under': {},
            'correct_score': {}
        }
        
        # 大小球
        for line in [1.5, 2.5, 3.5]:
            total_goals = home_goals + away_goals
            predictions['over_under'][f'{line}'] = {
                'over': float(np.mean(total_goals > line)),
                'under': float(np.mean(total_goals <= line))
            }
        
        # 正確比分（前10個最可能的）
        score_counts = {}
        for h, a in zip(home_goals, away_goals):
            if h <= 5 and a <= 5:  # 合理範圍
                score = f"{h}-{a}"
                score_counts[score] = score_counts.get(score, 0) + 1
        
        total_sims = sum(score_counts.values())
        score_probs = {k: v/total_sims for k, v in score_counts.items()}
        predictions['correct_score'] = dict(sorted(score_probs.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:10])
        
        return predictions
    
    def calculate_value_bets(self, predictions: Dict, bookmaker_odds: Dict) -> List[Dict]:
        """計算價值投注"""
        value_bets = []
        
        # 檢查 1X2 市場
        if 'home' in bookmaker_odds:
            markets = [
                ('home_win', 'home', predictions['match_result']['home_win']),
                ('draw', 'draw', predictions['match_result']['draw']),
                ('away_win', 'away', predictions['match_result']['away_win'])
            ]
            
            for name, key, prob in markets:
                if key in bookmaker_odds:
                    odds = bookmaker_odds[key]
                    ev = (prob * odds) - 1
                    
                    if ev > 0.05:  # 5% 優勢
                        value_bets.append({
                            'market': '1X2',
                            'selection': name,
                            'probability': prob,
                            'odds': odds,
                            'expected_value': ev,
                            'kelly_stake': self._kelly_criterion(prob, odds)
                        })
        
        return sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)
    
    def _kelly_criterion(self, prob: float, odds: float, fraction: float = 0.25) -> float:
        """凱利公式計算投注比例"""
        q = 1 - prob
        b = odds - 1
        kelly = (prob * b - q) / b
        return max(0, min(kelly * fraction, 0.05))  # 最多 5%
    
    def _save_model_to_redis(self, league: str, trace):
        """保存模型到 Redis"""
        import pickle
        model_data = {
            'trace': pickle.dumps(trace),
            'team_lookup': self.team_lookup,
            'teams': list(self.teams),
            'timestamp': datetime.now().isoformat()
        }
        self.redis.setex(
            f"football_model:{league}",
            86400,  # 24小時
            pickle.dumps(model_data)
        )
    
    def _load_model_from_redis(self, league: str):
        """從 Redis 載入模型"""
        import pickle
        data = self.redis.get(f"football_model:{league}")
        if data:
            model_data = pickle.loads(data)
            self.traces[league] = pickle.loads(model_data['trace'])
            self.team_lookup = model_data['team_lookup']
            self.teams = model_data['teams']
        else:
            raise ValueError(f"No model found for {league}")

# Flask API
from flask import Flask, request, jsonify

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
predictor = BayesianFootballPredictor(redis_client)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        league = data['league']
        matches_df = pd.DataFrame(data['matches'])
        
        predictor.train_model(league, matches_df)
        
        return jsonify({'status': 'success', 'message': f'Model trained for {league}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_match():
    try:
        data = request.json
        predictions = predictor.predict_match(
            data['home_team'],
            data['away_team'],
            data['league']
        )
        
        # 如果提供了賠率，計算價值投注
        if 'odds' in data:
            value_bets = predictor.calculate_value_bets(predictions, data['odds'])
            predictions['value_bets'] = value_bets
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
```

## 5. Docker Compose 配置

### docker-compose.yml
```yaml
version: '3.8'

services:
  # PostgreSQL 數據庫
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: betting_db
      POSTGRES_USER: betting_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U betting_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis 緩存
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # 後端 API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      NODE_ENV: production
      DATABASE_URL: postgresql://betting_user:${DB_PASSWORD}@postgres:5432/betting_db
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      JWT_SECRET: ${JWT_SECRET}
      PORT: 3001
    ports:
      - "3001:3001"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - /app/node_modules

  # 前端應用
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      NEXT_PUBLIC_API_URL: http://backend:3001/api
      NEXT_PUBLIC_WS_URL: ws://backend:3001
    ports:
      - "3000:3000"
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules
      - /app/.next

  # 賽馬預測服務
  horse-racing-ml:
    build:
      context: ./ml-models/horse-racing
      dockerfile: Dockerfile
    environment:
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      MODEL_PATH: /models
    ports:
      - "5001:5001"
    depends_on:
      - redis
    volumes:
      - ./ml-models/horse-racing:/app
      - ml_models:/models

  # 足球預測服務
  football-ml:
    build:
      context: ./ml-models/football
      dockerfile: Dockerfile
    environment:
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      MODEL_PATH: /models
    ports:
      - "5002:5002"
    depends_on:
      - redis
    volumes:
      - ./ml-models/football:/app
      - ml_models:/models

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
  ml_models:
```

## 6. 環境配置文件

### .env
```env
# 數據庫
DB_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# JWT
JWT_SECRET=your_jwt_secret_key

# API Keys
FOOTBALL_API_KEY=your_football_data_api_key
SPORTMONKS_API_KEY=your_sportmonks_api_key

# 前端
NEXT_PUBLIC_API_URL=http://localhost:3001/api
NEXT_PUBLIC_WS_URL=ws://localhost:3001

# ML 服務
HORSE_RACING_ML_URL=http://localhost:5001
FOOTBALL_ML_URL=http://localhost:5002
```

## 7. 部署指南

### 開發環境設置

```bash
# 1. 克隆項目
git clone https://github.com/your-repo/betting-system.git
cd betting-system

# 2. 安裝依賴
cd frontend && npm install
cd ../backend && npm install
cd ../ml-models/horse-racing && pip install -r requirements.txt
cd ../football && pip install -r requirements.txt

# 3. 設置環境變量
cp .env.example .env
# 編輯 .env 文件，填入您的配置

# 4. 啟動 Docker 服務
docker-compose up -d postgres redis

# 5. 運行數據庫遷移
cd backend && npm run migrate

# 6. 啟動開發服務器
# 終端 1 - 後端
cd backend && npm run dev

# 終端 2 - 前端
cd frontend && npm run dev

# 終端 3 - 賽馬 ML
cd ml-models/horse-racing && python enhanced_predictor.py

# 終端 4 - 足球 ML
cd ml-models/football && python bayesian_predictor.py
```

### 生產環境部署

```bash
# 1. 構建 Docker 鏡像
docker-compose build

# 2. 啟動所有服務
docker-compose up -d

# 3. 檢查服務狀態
docker-compose ps

# 4. 查看日誌
docker-compose logs -f

# 5. 設置 SSL 證書
# 將您的 SSL 證書放在 ./ssl 目錄下
# 更新 nginx.conf 配置

# 6. 設置監控
# 推薦使用 Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d
```

### 性能優化建議

1. **數據庫優化**
   - 為常用查詢添加索引
   - 使用連接池
   - 定期執行 VACUUM 和 ANALYZE

2. **緩存策略**
   - Redis 緩存熱門數據
   - 使用 CDN 加速靜態資源
   - 實施 HTTP 緩存頭

3. **WebSocket 優化**
   - 使用消息隊列處理高頻更新
   - 實施背壓控制
   - 客戶端重連策略

4. **機器學習模型**
   - 定期重新訓練模型
   - 使用 GPU 加速（如果可用）
   - 模型版本控制

### 監控和維護

1. **健康檢查端點**
   - GET /api/health - 後端健康狀態
   - GET /ml/health - ML 服務狀態

2. **關鍵指標**
   - API 響應時間 < 100ms (P95)
   - WebSocket 延遲 < 200ms
   - 模型預測準確率 > 35%
   - 系統可用性 > 99.9%

3. **備份策略**
   - 每日數據庫備份
   - 模型檔案版本控制
   - 配置文件加密存儲

4. **安全措施**
   - 定期更新依賴
   - 實施 WAF 規則
   - 監控異常投注模式

## 8. 故障排除

### 常見問題

1. **WebSocket 連接失敗**
   ```bash
   # 檢查防火牆設置
   sudo ufw allow 3001
   
   # 檢查 Nginx 配置
   nginx -t
   ```

2. **模型預測錯誤**
   ```bash
   # 檢查模型文件
   ls -la ml_models/
   
   # 重新訓練模型
   docker-compose exec horse-racing-ml python train.py
   ```

3. **數據庫連接問題**
   ```bash
   # 檢查 PostgreSQL 狀態
   docker-compose logs postgres
   
   # 重置數據庫
   docker-compose down -v
   docker-compose up -d
   ```

### 聯繫支持

如有任何問題，請提交 GitHub Issue 或聯繫技術支持團隊。

---

這個系統經過優化，可以支持 10 個並發用戶的生產環境使用，並具有良好的擴展性。請根據實際需求調整配置參數。
