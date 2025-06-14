# Dockerfile - 主應用程式
FROM node:18-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# 依賴安裝階段
FROM base AS deps
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force
RUN cp -R node_modules prod_node_modules
RUN npm ci && npm cache clean --force

# 構建階段
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# 生產階段
FROM base AS production
ENV NODE_ENV=production

# 創建非 root 用戶
RUN addgroup -g 1001 -S nodejs && adduser -S betting -u 1001

# 複製必要文件
COPY --from=deps --chown=betting:nodejs /app/prod_node_modules ./node_modules
COPY --from=builder --chown=betting:nodejs /app/dist ./dist
COPY --chown=betting:nodejs package*.json ./

# 創建必要目錄
RUN mkdir -p logs uploads && chown -R betting:nodejs logs uploads

USER betting

EXPOSE 3000 8080

CMD ["node", "dist/server.js"]
