# bookmaker_scraper.py - 生產級博彩公司爬蟲系統
import asyncio
import aiohttp
from aiohttp import TCPConnector
import json
import time
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from aiolimiter import AsyncLimiter
import redis
from fake_useragent import UserAgent
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import cloudscraper

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OddsData:
    """賠率數據結構"""
    bookmaker: str
    event_id: str
    market_type: str
    selection: str
    odds: float
    timestamp: datetime
    is_live: bool = False
    volume: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'bookmaker': self.bookmaker,
            'event_id': self.event_id,
            'market_type': self.market_type,
            'selection': self.selection,
            'odds': self.odds,
            'timestamp': self.timestamp.isoformat(),
            'is_live': self.is_live,
            'volume': self.volume
        }

class ProxyManager:
    """代理管理器"""
    
    def __init__(self, proxy_list_path: str = 'proxies.txt'):
        self.proxies = self._load_proxies(proxy_list_path)
        self.failed_proxies = set()
        self.proxy_stats = {}
        
    def _load_proxies(self, path: str) -> List[str]:
        """載入代理列表"""
        try:
            with open(path, 'r') as f:
                proxies = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(proxies)} proxies")
            return proxies
        except FileNotFoundError:
            logger.warning("Proxy file not found, using direct connection")
            return []
    
    async def get_working_proxy(self) -> Optional[str]:
        """獲取可用的代理"""
        available = [p for p in self.proxies if p not in self.failed_proxies]
        
        if not available:
            # 重置失敗列表
            self.failed_proxies.clear()
            available = self.proxies
        
        if not available:
            return None
        
        # 選擇成功率最高的代理
        if self.proxy_stats:
            sorted_proxies = sorted(
                available,
                key=lambda p: self.proxy_stats.get(p, {}).get('success_rate', 0.5),
                reverse=True
            )
            return sorted_proxies[0]
        
        return random.choice(available)
    
    def mark_proxy_failed(self, proxy: str):
        """標記代理失敗"""
        self.failed_proxies.add(proxy)
        
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {'success': 0, 'fail': 0}
        
        self.proxy_stats[proxy]['fail'] += 1
        self._update_success_rate(proxy)
    
    def mark_proxy_success(self, proxy: str):
        """標記代理成功"""
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {'success': 0, 'fail': 0}
        
        self.proxy_stats[proxy]['success'] += 1
        self._update_success_rate(proxy)
    
    def _update_success_rate(self, proxy: str):
        """更新代理成功率"""
        stats = self.proxy_stats[proxy]
        total = stats['success'] + stats['fail']
        if total > 0:
            stats['success_rate'] = stats['success'] / total

class BookmakerScraper(ABC):
    """博彩公司爬蟲基類"""
    
    def __init__(self, name: str, proxy_manager: Optional[ProxyManager] = None):
        self.name = name
        self.proxy_manager = proxy_manager
        self.rate_limiter = AsyncLimiter(10, 60)  # 每分鐘10次請求
        self.session = None
        self.browser = None
        self.ua = UserAgent()
        
    @abstractmethod
    async def scrape_odds(self, event_id: str) -> List[OddsData]:
        """爬取賠率數據（子類實現）"""
        pass
    
    async def get_session(self) -> aiohttp.ClientSession:
        """獲取或創建會話"""
        if self.session is None or self.session.closed:
            connector = TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                ssl=False
            )
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': self.ua.random,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
        
        return self.session
    
    async def make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[str]:
        """發起HTTP請求"""
        async with self.rate_limiter:
            proxy = None
            if self.proxy_manager:
                proxy = await self.proxy_manager.get_working_proxy()
            
            try:
                session = await self.get_session()
                
                async with session.request(
                    method, 
                    url, 
                    proxy=proxy,
                    **kwargs
                ) as response:
                    if response.status == 200:
                        if proxy and self.proxy_manager:
                            self.proxy_manager.mark_proxy_success(proxy)
                        return await response.text()
                    else:
                        logger.warning(f"{self.name}: HTTP {response.status} for {url}")
                        if proxy and self.proxy_manager:
                            self.proxy_manager.mark_proxy_failed(proxy)
                        return None
                        
            except Exception as e:
                logger.error(f"{self.name}: Request error: {str(e)}")
                if proxy and self.proxy_manager:
                    self.proxy_manager.mark_proxy_failed(proxy)
                return None
    
    async def close(self):
        """關閉連接"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        if self.browser:
            await self.browser.close()

class Bet365Scraper(BookmakerScraper):
    """Bet365 爬蟲"""
    
    def __init__(self, proxy_manager: Optional[ProxyManager] = None):
        super().__init__("Bet365", proxy_manager)
        self.base_url = "https://www.bet365.com"
        
    async def scrape_odds(self, event_id: str) -> List[OddsData]:
        """爬取 Bet365 賠率"""
        odds_data = []
        
        # 需要使用 Playwright 處理動態內容
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--disable-web-security',
                    '--disable-features=BlockInsecurePrivateNetworkRequests',
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self.ua.random,
                ignore_https_errors=True,
                java_script_enabled=True
            )
            
            # 注入反檢測腳本
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                window.chrome = {
                    runtime: {}
                };
                
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({
                        query: () => Promise.resolve({ state: 'granted' })
                    })
                });
            """)
            
            page = await context.new_page()
            
            try:
                # 導航到賽事頁面
                event_url = f"{self.base_url}/sports/event/{event_id}"
                await page.goto(event_url, wait_until='networkidle')
                
                # 等待賠率加載
                await page.wait_for_selector('.bet-odds', timeout=10000)
                
                # 提取賠率數據
                odds_elements = await page.query_selector_all('.odds-row')
                
                for element in odds_elements:
                    try:
                        market_type = await element.query_selector('.market-name')
                        market_type_text = await market_type.inner_text() if market_type else 'Unknown'
                        
                        selections = await element.query_selector_all('.selection')
                        
                        for selection in selections:
                            selection_name = await selection.query_selector('.selection-name')
                            selection_name_text = await selection_name.inner_text() if selection_name else 'Unknown'
                            
                            odds_value = await selection.query_selector('.odds-value')
                            odds_value_text = await odds_value.inner_text() if odds_value else '0'
                            
                            try:
                                odds_float = float(odds_value_text)
                            except ValueError:
                                continue
                            
                            odds_data.append(OddsData(
                                bookmaker=self.name,
                                event_id=event_id,
                                market_type=self._normalize_market_type(market_type_text),
                                selection=selection_name_text,
                                odds=odds_float,
                                timestamp=datetime.now()
                            ))
                    
                    except Exception as e:
                        logger.error(f"Error parsing odds element: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Bet365 scraping error: {str(e)}")
            
            finally:
                await browser.close()
        
        return odds_data
    
    def _normalize_market_type(self, market_type: str) -> str:
        """標準化市場類型"""
        market_map = {
            'match result': '1X2',
            'full time result': '1X2',
            'both teams to score': 'BTTS',
            'over/under': 'O/U',
            'asian handicap': 'AH'
        }
        
        normalized = market_type.lower().strip()
        return market_map.get(normalized, normalized)

class WilliamHillScraper(BookmakerScraper):
    """William Hill 爬蟲"""
    
    def __init__(self, proxy_manager: Optional[ProxyManager] = None):
        super().__init__("WilliamHill", proxy_manager)
        self.base_url = "https://sports.williamhill.com"
        
    async def scrape_odds(self, event_id: str) -> List[OddsData]:
        """爬取 William Hill 賠率"""
        odds_data = []
        
        # 構建API URL
        api_url = f"{self.base_url}/api/v1/events/{event_id}/markets"
        
        response = await self.make_request(api_url)
        
        if response:
            try:
                data = json.loads(response)
                
                for market in data.get('markets', []):
                    market_type = market.get('type', 'Unknown')
                    
                    for selection in market.get('selections', []):
                        odds_data.append(OddsData(
                            bookmaker=self.name,
                            event_id=event_id,
                            market_type=self._normalize_market_type(market_type),
                            selection=selection.get('name', 'Unknown'),
                            odds=float(selection.get('price', {}).get('decimal', 0)),
                            timestamp=datetime.now()
                        ))
                        
            except json.JSONDecodeError as e:
                logger.error(f"William Hill JSON decode error: {str(e)}")
            except Exception as e:
                logger.error(f"William Hill parsing error: {str(e)}")
        
        return odds_data
    
    def _normalize_market_type(self, market_type: str) -> str:
        """標準化市場類型"""
        market_map = {
            'match_betting': '1X2',
            'btts': 'BTTS',
            'total_goals': 'O/U',
            'handicap': 'AH'
        }
        
        normalized = market_type.lower().strip()
        return market_map.get(normalized, normalized)

class PinnacleScraper(BookmakerScraper):
    """Pinnacle 爬蟲（使用API）"""
    
    def __init__(self, api_key: str, proxy_manager: Optional[ProxyManager] = None):
        super().__init__("Pinnacle", proxy_manager)
        self.api_key = api_key
        self.base_url = "https://api.pinnacle.com/v3"
        
    async def scrape_odds(self, event_id: str) -> List[OddsData]:
        """爬取 Pinnacle 賠率"""
        odds_data = []
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        
        # 獲取賠率
        odds_url = f"{self.base_url}/odds"
        params = {
            'eventIds': event_id,
            'oddsFormat': 'decimal'
        }
        
        response = await self.make_request(
            odds_url,
            headers=headers,
            params=params
        )
        
        if response:
            try:
                data = json.loads(response)
                
                for league in data.get('leagues', []):
                    for event in league.get('events', []):
                        if str(event.get('id')) == event_id:
                            for period in event.get('periods', []):
                                # 1X2 市場
                                if 'moneyline' in period:
                                    ml = period['moneyline']
                                    
                                    if 'home' in ml:
                                        odds_data.append(OddsData(
                                            bookmaker=self.name,
                                            event_id=event_id,
                                            market_type='1X2',
                                            selection='HOME',
                                            odds=ml['home'],
                                            timestamp=datetime.now()
                                        ))
                                    
                                    if 'draw' in ml:
                                        odds_data.append(OddsData(
                                            bookmaker=self.name,
                                            event_id=event_id,
                                            market_type='1X2',
                                            selection='DRAW',
                                            odds=ml['draw'],
                                            timestamp=datetime.now()
                                        ))
                                    
                                    if 'away' in ml:
                                        odds_data.append(OddsData(
                                            bookmaker=self.name,
                                            event_id=event_id,
                                            market_type='1X2',
                                            selection='AWAY',
                                            odds=ml['away'],
                                            timestamp=datetime.now()
                                        ))
                                
                                # 大小盤
                                if 'totals' in period:
                                    for total in period['totals']:
                                        line = total.get('points', 0)
                                        
                                        if 'over' in total:
                                            odds_data.append(OddsData(
                                                bookmaker=self.name,
                                                event_id=event_id,
                                                market_type='O/U',
                                                selection=f'OVER_{line}',
                                                odds=total['over'],
                                                timestamp=datetime.now()
                                            ))
                                        
                                        if 'under' in total:
                                            odds_data.append(OddsData(
                                                bookmaker=self.name,
                                                event_id=event_id,
                                                market_type='O/U',
                                                selection=f'UNDER_{line}',
                                                odds=total['under'],
                                                timestamp=datetime.now()
                                            ))
                                
                                # 亞洲盤
                                if 'spreads' in period:
                                    for spread in period['spreads']:
                                        hdp = spread.get('hdp', 0)
                                        
                                        if 'home' in spread:
                                            odds_data.append(OddsData(
                                                bookmaker=self.name,
                                                event_id=event_id,
                                                market_type='AH',
                                                selection=f'HOME_{hdp}',
                                                odds=spread['home'],
                                                timestamp=datetime.now()
                                            ))
                                        
                                        if 'away' in spread:
                                            odds_data.append(OddsData(
                                                bookmaker=self.name,
                                                event_id=event_id,
                                                market_type='AH',
                                                selection=f'AWAY_{-hdp}',
                                                odds=spread['away'],
                                                timestamp=datetime.now()
                                            ))
                
            except Exception as e:
                logger.error(f"Pinnacle parsing error: {str(e)}")
        
        return odds_data

class OddsCollectionOrchestrator:
    """賠率收集協調器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.proxy_manager = ProxyManager()
        self.scrapers = self._initialize_scrapers()
        self.failed_attempts = {}
        
    def _initialize_scrapers(self) -> Dict[str, BookmakerScraper]:
        """初始化所有爬蟲"""
        scrapers = {
            'bet365': Bet365Scraper(self.proxy_manager),
            'williamhill': WilliamHillScraper(self.proxy_manager),
            # 'pinnacle': PinnacleScraper(api_key='your_api_key', proxy_manager=self.proxy_manager),
            # 添加更多博彩公司...
        }
        
        logger.info(f"Initialized {len(scrapers)} scrapers")
        return scrapers
    
    async def collect_odds_for_event(self, event_id: str) -> List[OddsData]:
        """收集單個賽事的所有賠率"""
        all_odds = []
        
        # 並發收集所有博彩公司的賠率
        tasks = []
        for name, scraper in self.scrapers.items():
            # 檢查是否需要重試
            if self._should_retry(name, event_id):
                task = self._collect_with_retry(scraper, event_id)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_odds.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Collection error: {str(result)}")
        
        # 存儲到 Redis
        await self._store_odds_to_redis(event_id, all_odds)
        
        return all_odds
    
    async def _collect_with_retry(self, scraper: BookmakerScraper, event_id: str, max_retries: int = 3) -> List[OddsData]:
        """帶重試的賠率收集"""
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f"Collecting odds from {scraper.name} for event {event_id}")
                odds = await scraper.scrape_odds(event_id)
                
                if odds:
                    # 成功，重置失敗計數
                    self.failed_attempts.pop(f"{scraper.name}:{event_id}", None)
                    return odds
                
            except Exception as e:
                logger.error(f"{scraper.name} collection error (attempt {retry_count + 1}): {str(e)}")
            
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(2 ** retry_count)  # 指數退避
        
        # 記錄失敗
        self._record_failure(scraper.name, event_id)
        return []
    
    def _should_retry(self, bookmaker: str, event_id: str) -> bool:
        """判斷是否應該重試"""
        key = f"{bookmaker}:{event_id}"
        
        if key not in self.failed_attempts:
            return True
        
        last_attempt = self.failed_attempts[key]['last_attempt']
        fail_count = self.failed_attempts[key]['count']
        
        # 根據失敗次數決定重試間隔
        retry_interval = min(300 * (2 ** fail_count), 3600)  # 最多1小時
        
        return (datetime.now() - last_attempt).total_seconds() > retry_interval
    
    def _record_failure(self, bookmaker: str, event_id: str):
        """記錄失敗"""
        key = f"{bookmaker}:{event_id}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {'count': 0, 'last_attempt': datetime.now()}
        
        self.failed_attempts[key]['count'] += 1
        self.failed_attempts[key]['last_attempt'] = datetime.now()
    
    async def _store_odds_to_redis(self, event_id: str, odds_data: List[OddsData]):
        """存儲賠率到 Redis"""
        pipeline = self.redis_client.pipeline()
        
        for odds in odds_data:
            key = f"odds:{event_id}:{odds.bookmaker}:{odds.market_type}:{odds.selection}"
            value = json.dumps(odds.to_dict())
            
            # 設置過期時間（5分鐘）
            pipeline.setex(key, 300, value)
            
            # 發布到頻道
            pipeline.publish('odds:updates', value)
        
        await pipeline.execute()
        logger.info(f"Stored {len(odds_data)} odds for event {event_id}")
    
    async def run_continuous_collection(self, event_ids: List[str], interval: int = 30):
        """持續收集賠率"""
        logger.info(f"Starting continuous collection for {len(event_ids)} events")
        
        while True:
            start_time = time.time()
            
            # 並發收集所有賽事
            tasks = [self.collect_odds_for_event(event_id) for event_id in event_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 計算下次收集時間
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            
            logger.info(f"Collection cycle completed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
    
    async def close(self):
        """關閉所有連接"""
        tasks = [scraper.close() for scraper in self.scrapers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

# 反檢測工具類
class AntiDetectionHelper:
    """反檢測輔助類"""
    
    @staticmethod
    def get_random_headers() -> Dict[str, str]:
        """獲取隨機請求頭"""
        ua = UserAgent()
        
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice([
                'en-US,en;q=0.9',
                'en-GB,en;q=0.9',
                'en-US,en;q=0.8,zh-CN;q=0.6',
                'en-US,en;q=0.9,es;q=0.8'
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # 隨機添加一些額外的頭
        if random.random() > 0.5:
            headers['Referer'] = 'https://www.google.com/'
        
        if random.random() > 0.7:
            headers['X-Forwarded-For'] = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
        
        return headers
    
    @staticmethod
    def add_random_delay(min_delay: float = 0.5, max_delay: float = 3.0):
        """添加隨機延遲"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    @staticmethod
    def generate_mouse_movements() -> List[Tuple[int, int]]:
        """生成模擬鼠標移動軌跡"""
        movements = []
        
        # 起始點
        x, y = random.randint(100, 500), random.randint(100, 500)
        
        # 生成貝塞爾曲線軌跡
        for _ in range(random.randint(5, 15)):
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            
            x = max(0, min(1920, x + dx))
            y = max(0, min(1080, y + dy))
            
            movements.append((x, y))
        
        return movements

# 主程序
async def main():
    """主程序入口"""
    # 初始化 Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        decode_responses=True
    )
    
    # 創建協調器
    orchestrator = OddsCollectionOrchestrator(redis_client)
    
    # 測試賽事ID列表
    event_ids = ['12345', '67890', '11111']
    
    try:
        # 開始持續收集
        await orchestrator.run_continuous_collection(event_ids, interval=30)
    except KeyboardInterrupt:
        logger.info("Stopping collection...")
    finally:
        await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(main())