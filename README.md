# Advanced Betting Platform

## Overview

This project is a comprehensive betting system designed to offer a rich user experience for sports betting. Core functionalities include real-time odds display, arbitrage opportunity detection, and machine learning-based predictions for horse racing and football events. The platform is architected as a set of microservices to ensure scalability and maintainability.

## Project Structure

The project has been organized into the following main directories:

*   **`frontend/`**: Contains the Next.js 14 + TypeScript frontend application, including UI components built with Tremor.
*   **`backend/services/`**: Houses the backend microservices.
    *   `api_gateway/`: The main API Gateway handling client requests and routing to other services.
    *   `arbitrage_engine/`: Service responsible for detecting arbitrage opportunities.
    *   `monitoring_service/`: Service for system monitoring and alerts.
*   **`ml_models/`**: Contains the machine learning models and their serving applications.
    *   `horse_racing/`: Horse racing prediction model and service.
    *   `football/`: Football match outcome prediction model (Bayesian) and service.
*   **`scrapers/`**: Includes the web scraper services for collecting odds and other data.
    *   `bookmaker_scraper/`: Scraper for fetching data from bookmaker websites.
*   **`database/`**:
    *   `init_scripts/`: SQL scripts for initial database schema setup (e.g., `001_initial_schema.sql`).
*   **`deployment_configs/`**: Contains configuration files for various deployment environments.
    *   `docker/`: Docker Compose configurations for local development and testing.
    *   `kubernetes/`: Basic Kubernetes manifest templates.
    *   `nginx/`: Nginx configuration for reverse proxy and load balancing.
*   **`logs/`**: Intended for storing log files (placeholder, actual log output might be service-specific or managed by Docker/K8s).
*   **`scripts/`**: Utility scripts for development, deployment, or maintenance tasks (placeholder).

## Initial Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Configure Environment:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and update the placeholder values with your actual configurations (database credentials, API keys, secrets, etc.).

3.  **Essential Services (Docker Compose):**
    *   To start essential services like the database (PostgreSQL/TimescaleDB) and Redis, use:
        ```bash
        docker-compose -f deployment_configs/docker/docker-compose.yml up -d timescaledb redis
        ```
    *   Wait for these services to be healthy before starting dependent application services.

## Running Components for Development

Below are general guidelines for running individual services. Ensure all dependencies (as defined in `.env`) are available and accessible.

*   **Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```
    (The frontend will typically be available at `http://localhost:3001` if the default Next.js port is 3000 and it's adjusted, or `http://localhost:3000`.)

*   **Backend - API Gateway:**
    ```bash
    cd backend/services/api_gateway
    # Ensure package.json is complete and accurate
    npm install
    npm run dev # Or: node index.js (depends on package.json scripts)
    ```

*   **Backend - Other Node.js Services (Arbitrage, Monitoring):**
    ```bash
    cd backend/services/<service_name> # e.g., arbitrage_engine
    # Ensure package.json is complete and accurate
    npm install
    node index.js # Or a specific start script
    ```

*   **Python ML Models & Scraper Services:**
    ```bash
    cd <service_directory> # e.g., ml_models/football/ or scrapers/bookmaker_scraper/
    # Ensure requirements.txt is complete and accurate
    pip install -r requirements.txt
    python <script_name>.py # e.g., bayesian_predictor.py or scraper.py
    ```
    (Note: ML models with Flask APIs are typically run with Gunicorn in Docker, e.g., `gunicorn --bind 0.0.0.0:5001 bayesian_predictor:app`)

## Key Technologies

*   **Frontend:** Next.js 14, React, TypeScript, Tremor UI, Tailwind CSS
*   **Backend:** Node.js, Express.js (implied for Node.js services)
*   **Machine Learning:** Python, PyMC3 (for Football ML), Scikit-learn, XGBoost, LightGBM, PyTorch (intended for Horse Racing ML)
*   **Database:** PostgreSQL with TimescaleDB extension
*   **Caching/Messaging:** Redis
*   **Message Queue:** Apache Kafka
*   **Containerization & Orchestration:** Docker, Docker Compose, Kubernetes (basic templates)
*   **Reverse Proxy:** Nginx
*   **Web Scraping:** Playwright (Python)

## Current Status & Important Notes

This project has undergone a significant restructuring to a microservice-based architecture.

*   **Backend Services:** The core logic for the `api_gateway`, `arbitrage_engine`, and `monitoring_service` is based on the JavaScript files initially provided for a monolithic system and has been moved into the respective service directories. Placeholder `package.json` files have been added; these will require verification and completion of dependencies based on the actual code within each service.
*   **Frontend:** The frontend application structure has been created. Core files (`package.json`, `layout.tsx`, `page.tsx`, `LiveOddsPanel.tsx`, `useBettingWebSocket.ts`, `horse-racing/page.tsx`, `services/api.ts`) have been populated with more detailed code based on (simulated) extraction from user-provided markdown. Other UI components and pages are currently placeholders and require full implementation of their business logic, data fetching, and UI details.
*   **Football ML Model (`ml_models/football/bayesian_predictor.py`):** The code for this service was populated based on (simulated) extraction from user-provided markdown, including a Flask API structure and a `requirements.txt` file.
*   **Horse Racing ML Model (`ml_models/horse_racing/predictor.py`):** This directory contains the *original* Python script. Attempts to update it to an "enhanced" version (with ensemble models) failed due to tooling limitations during the refactoring process. A `requirements.txt` file has been generated based on the imports of this existing script.
*   **Scraper Service (`scrapers/bookmaker_scraper/scraper.py`):** Contains the original Python scraper script. A placeholder `requirements.txt` has been added and will need verification.

**Further Work:**
Significant effort is required to:
*   Implement the detailed business logic and UI for the frontend components.
*   Finalize and verify dependencies in all `package.json` and `requirements.txt` files.
*   Thoroughly test inter-service communication (API calls, Kafka messaging, Redis usage).
*   Develop and test robust data ingestion pipelines for ML models and scrapers.
*   Implement comprehensive error handling, logging, and security measures across all services.
*   Conduct thorough unit, integration, and end-to-end testing.
*   Expand and refine deployment configurations (Docker, Kubernetes) for production readiness.

This README provides a snapshot of the project post-restructuring. It is a foundational step towards a fully operational system.

---
## 繁體中文 (Traditional Chinese)

# 高階投注平台

## 總覽

本專案為一綜合性投注系統，旨在為體育博彩提供豐富的使用者體驗。核心功能包括即時賠率顯示、套利機會偵測，以及基於機器學習的賽馬和足球賽事預測。本平台採用微服務架構，以確保可擴展性和可維護性。

## 專案結構

本專案已組織為以下主要目錄：

*   **`frontend/`**: 包含 Next.js 14 + TypeScript 前端應用程式，以及使用 Tremor 建構的 UI 組件。
*   **`backend/services/`**: 存放後端微服務。
    *   `api_gateway/`: 主要的 API 閘道，處理客戶端請求並路由至其他服務。
    *   `arbitrage_engine/`: 負責偵測套利機會的服務。
    *   `monitoring_service/`: 用於系統監控和警報的服務。
*   **`ml_models/`**: 包含機器學習模型及其服務應用程式。
    *   `horse_racing/`: 賽馬預測模型及服務。
    *   `football/`: 足球賽果預測模型（貝葉斯）及服務。
*   **`scrapers/`**: 包括用於收集賠率和其他數據的網路爬蟲服務。
    *   `bookmaker_scraper/`: 從博彩公司網站擷取數據的爬蟲。
*   **`database/`**:
    *   `init_scripts/`: 用於初始資料庫結構設定的 SQL 指令碼（例如 `001_initial_schema.sql`）。
*   **`deployment_configs/`**: 包含適用於各種部署環境的設定檔。
    *   `docker/`: Docker Compose 設定，用於本機開發和測試。
    *   `kubernetes/`: 基本的 Kubernetes 清單範本。
    *   `nginx/`: Nginx 設定，用於反向代理和負載平衡。
*   **`logs/`**: 設計用於儲存日誌檔案（佔位符，實際日誌輸出可能針對特定服務或由 Docker/K8s 管理）。
*   **`scripts/`**: 用於開發、部署或維護任務的工具指令碼（佔位符）。

## 初始設定

1.  **複製儲存庫：**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **設定環境：**
    *   複製環境變數範例檔案：
        ```bash
        cp .env.example .env
        ```
    *   開啟 `.env` 檔案，並使用您的實際設定（資料庫憑證、API 金鑰、密鑰等）更新佔位符值。

3.  **基礎服務 (Docker Compose)：**
    *   若要啟動基礎服務如資料庫 (PostgreSQL/TimescaleDB) 和 Redis，請使用：
        ```bash
        docker-compose -f deployment_configs/docker/docker-compose.yml up -d timescaledb redis
        ```
    *   在啟動相依的應用程式服務之前，請等待這些服務進入健康狀態。

## 開發模式下執行個別組件

以下為執行個別服務的一般指南。請確保 `.env` 中定義的所有相依性均可用且可存取。

*   **前端 (Frontend)：**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```
    （前端通常可在 `http://localhost:3001` 存取，如果 Next.js 預設埠為 3000 並已調整，或 `http://localhost:3000`。）

*   **後端 - API 閘道 (Backend - API Gateway)：**
    ```bash
    cd backend/services/api_gateway
    # 確認 package.json 完整且正確
    npm install
    npm run dev # 或：node index.js (取決於 package.json 中的指令碼)
    ```

*   **後端 - 其他 Node.js 服務 (套利、監控)：**
    ```bash
    cd backend/services/<service_name> # 例如 arbitrage_engine
    # 確認 package.json 完整且正確
    npm install
    node index.js # 或特定的啟動指令碼
    ```

*   **Python 機器學習模型 & 爬蟲服務：**
    ```bash
    cd <service_directory> # 例如 ml_models/football/ 或 scrapers/bookmaker_scraper/
    # 確認 requirements.txt 完整且正確
    pip install -r requirements.txt
    python <script_name>.py # 例如 bayesian_predictor.py 或 scraper.py
    ```
    （註：具有 Flask API 的機器學習模型通常在 Docker 中使用 Gunicorn 執行，例如：`gunicorn --bind 0.0.0.0:5001 bayesian_predictor:app`）

## 主要技術棧

*   **前端 (Frontend)：** Next.js 14, React, TypeScript, Tremor UI, Tailwind CSS
*   **後端 (Backend)：** Node.js, Express.js (隱含用於 Node.js 服務)
*   **機器學習 (Machine Learning)：** Python, PyMC3 (用於足球機器學習), Scikit-learn, XGBoost, LightGBM, PyTorch (預計用於賽馬機器學習)
*   **資料庫 (Database)：** PostgreSQL 搭配 TimescaleDB 擴充套件
*   **快取/訊息傳遞 (Caching/Messaging)：** Redis
*   **訊息佇列 (Message Queue)：** Apache Kafka
*   **容器化與編排 (Containerization & Orchestration)：** Docker, Docker Compose, Kubernetes (基本範本)
*   **反向代理 (Reverse Proxy)：** Nginx
*   **網路爬蟲 (Web Scraping)：** Playwright (Python)

## 目前狀態與重要注意事項

本專案已進行重大重組，轉為基於微服務的架構。

*   **後端服務：** `api_gateway`、`arbitrage_engine` 和 `monitoring_service` 的核心邏輯基於最初為單體系統提供的 JavaScript 檔案，現已移至相應的服務目錄中。已添加佔位符 `package.json` 檔案；這些檔案需要根據每個服務內的實際程式碼來驗證和補齊相依性。
*   **前端：** 前端應用程式結構已建立。核心檔案（`package.json`, `layout.tsx`, `page.tsx`, `LiveOddsPanel.tsx`, `useBettingWebSocket.ts`, `horse-racing/page.tsx`, `services/api.ts`）已根據使用者提供的 markdown 內容（模擬擷取）填入更詳細的程式碼。其他 UI 組件和頁面目前為佔位符，需要完整實作其業務邏輯、數據擷取和 UI 細節。
*   **足球機器學習模型 (`ml_models/football/bayesian_predictor.py`)：** 此服務的程式碼是根據使用者提供的 markdown 內容（模擬擷取）填入的，包括 Flask API 結構和 `requirements.txt` 檔案。
*   **賽馬機器學習模型 (`ml_models/horse_racing/predictor.py`)：** 此目錄包含 *原始的* Python 指令碼。由於重構過程中工具的限制，嘗試將其更新為「增強版」（包含集成模型）失敗。已根據此現有指令碼的匯入產生了 `requirements.txt` 檔案。
*   **爬蟲服務 (`scrapers/bookmaker_scraper/scraper.py`)：** 包含原始的 Python 爬蟲指令碼。已添加佔位符 `requirements.txt`，需要進行驗證。

**後續工作：**
需要大量投入以完成以下工作：
*   實作前端組件的詳細業務邏輯和 UI。
*   最終確認並驗證所有 `package.json` 和 `requirements.txt` 檔案中的相依性。
*   徹底測試服務間通訊（API 呼叫、Kafka 訊息傳遞、Redis 使用）。
*   為機器學習模型和爬蟲開發並測試穩健的數據擷取流程。
*   在所有服務中實施全面的錯誤處理、日誌記錄和安全措施。
*   進行徹底的單元測試、整合測試和端對端測試。
*   擴展和完善部署設定（Docker, Kubernetes），使其達到生產就緒狀態。

本 README 檔案提供了專案重組後的概況。這是朝向一個完全可運作系統的基礎步驟。
