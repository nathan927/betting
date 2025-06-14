import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios';
// import { useAuthStore } from '@/stores/authStore'; // Actual path for Zustand store

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// Queue for failed requests due to token refresh
let failedRequestQueue: Array<{
  resolve: (value?: any) => void;
  reject: (error?: any) => void;
  config: InternalAxiosRequestConfig;
}> = [];

let isRefreshingToken = false;

const processFailedQueue = (error: AxiosError | null, token: string | null = null) => {
  failedRequestQueue.forEach(prom => {
    if (error) {
      prom.reject(error);
    } else if (token && prom.config.headers) {
      prom.config.headers['Authorization'] = `Bearer ${token}`;
      apiClient(prom.config).then(prom.resolve).catch(prom.reject); // Retry the original request
    }
  });
  failedRequestQueue = [];
};

// Request Interceptor: Add JWT token to headers
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // const { token } = useAuthStore.getState(); // Get token from Zustand (or context/localStorage)
    const token = typeof window !== 'undefined' ? localStorage.getItem("dummy-jwt-token") : null; // Placeholder for actual token
    if (token && config.headers) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error: AxiosError) => Promise.reject(error)
);

// Response Interceptor: Handle token refresh logic
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && !originalRequest._retry && originalRequest.url !== '/auth/refresh-token') {
      if (isRefreshingToken) {
        return new Promise((resolve, reject) => {
          failedRequestQueue.push({ resolve, reject, config: originalRequest });
        });
      }

      originalRequest._retry = true;
      isRefreshingToken = true;

      // const { refreshToken, setAuth, clearAuth } = useAuthStore.getState();
      const storedRefreshToken = typeof window !== 'undefined' ? localStorage.getItem("dummy-refresh-token") : null; // Placeholder

      if (!storedRefreshToken) {
        isRefreshingToken = false;
        // clearAuth(); // Clear user session as no refresh token is available
        console.error("No refresh token available for token refresh.");
        processFailedQueue(error, null);
        return Promise.reject(error);
      }

      try {
        console.log("Attempting to refresh token...");
        const { data } = await axios.post(`${API_BASE_URL}/auth/refresh-token`, {
          refreshToken: storedRefreshToken,
        });

        const newAccessToken = data.accessToken;
        const newRefreshToken = data.refreshToken; // Assuming backend might issue a new refresh token

        // setAuth(newAccessToken, newRefreshToken, data.user); // Update store with new tokens and user data
        if (typeof window !== 'undefined') { // Placeholder token update
            localStorage.setItem("dummy-jwt-token", newAccessToken);
            if (newRefreshToken) localStorage.setItem("dummy-refresh-token", newRefreshToken);
        }

        console.log("Token refreshed successfully.");
        if (originalRequest.headers) {
          originalRequest.headers['Authorization'] = `Bearer ${newAccessToken}`;
        }
        processFailedQueue(null, newAccessToken); // Process queued requests with new token
        return apiClient(originalRequest); // Retry the original request with new token
      } catch (refreshError: any) {
        // clearAuth(); // Clear session if refresh fails
        console.error("Token refresh failed:", refreshError?.response?.data || refreshError.message);
        processFailedQueue(refreshError as AxiosError, null);
        // Redirect to login or show global error message
        // window.location.href = '/login';
        return Promise.reject(refreshError);
      } finally {
        isRefreshingToken = false;
      }
    }
    return Promise.reject(error);
  }
);

// --- API Service Functions ---

// Auth Service
export const authService = {
  login: (credentials: unknown) => apiClient.post('/auth/login', credentials),
  register: (userData: unknown) => apiClient.post('/auth/register', userData),
  logout: () => apiClient.post('/auth/logout'),
  refreshToken: (tokenData: unknown) => apiClient.post('/auth/refresh-token', tokenData),
  getProfile: () => apiClient.get('/users/me'),
  updateProfile: (profileData: unknown) => apiClient.put('/users/me', profileData),
};

// Betting Service
export const bettingService = {
  placeBet: (betData: unknown) => apiClient.post('/bets', betData),
  getBetHistory: (params?: unknown) => apiClient.get('/bets/history', { params }),
  getBetDetails: (betId: string) => apiClient.get(`/bets/${betId}`),
  cashOutBet: (betId: string) => apiClient.post(`/bets/${betId}/cashout`),
};

// Events & Odds Service
export const eventsService = {
  getSports: () => apiClient.get('/sports'),
  getEvents: (params?: unknown) => apiClient.get('/events', { params }), // e.g., { sportKey: 'football', date: 'YYYY-MM-DD', status: 'live' }
  getEventDetails: (eventId: string) => apiClient.get(`/events/${eventId}`),
  getMarketsForEvent: (eventId: string) => apiClient.get(`/events/${eventId}/markets`),
  getOddsForMarket: (marketId: string) => apiClient.get(`/markets/${marketId}/odds`), // If specific odds endpoint exists
};

// Prediction Services
export const predictionService = {
  // Horse Racing
  getHorseRaces: (params?: unknown) => apiClient.get('/predictions/horse-racing/races', { params }), // e.g., { date: 'YYYY-MM-DD', venue: 'HK_HV' }
  getHorseRacePrediction: (raceId: string) => apiClient.get(`/predictions/horse-racing/${raceId}`),
  getHorseDetails: (horseId: string) => apiClient.get(`/horses/${horseId}`), // Example additional endpoint

  // Football
  getFootballLeagues: () => apiClient.get('/predictions/football/leagues'),
  getFootballMatches: (params?: unknown) => apiClient.get('/predictions/football/matches', { params }), // e.g., { leagueId: 'ENG_PL', date: 'YYYY-MM-DD'}
  getFootballMatchPrediction: (matchId: string) => apiClient.get(`/predictions/football/${matchId}`),
};

// User Settings & Responsible Gaming
export const userSettingsService = {
  getLimits: () => apiClient.get('/users/me/limits'),
  setLimits: (limitsData: unknown) => apiClient.post('/users/me/limits', limitsData),
  getAccountActivity: (params?: unknown) => apiClient.get('/users/me/activity', { params }),
};

// --- Re-exporting for easier import from components ---
// This part matches the previous api.ts structure if components relied on these direct exports
export const { loginUser, registerUser, logoutUser } = authService; // Example, adjust as needed
export const { getRecentBets, placeBet, getBetDetails } = bettingService;
export const { getEvents, getEventDetails } = eventsService;
export const { getHorseRacingRaces, getHorseRacingPrediction } = predictionService;
export const { getFootballMatches, getFootballPrediction } = predictionService;


export default apiClient;
