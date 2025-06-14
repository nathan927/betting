import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
// import apiClient from '@/services/api'; // If login/logout directly call API methods

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  // Add other user properties as needed
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (token: string, refreshToken: string, userData: User) => void;
  logout: () => void;
  setToken: (token: string, refreshToken: string) => void;
  setUser: (userData: User) => void;
  clearAuth: () => void;
  loadAuthFromStorage: () => void; // Manually load if needed, though persist middleware handles it
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: (token, refreshToken, userData) => {
        set({
          token,
          refreshToken,
          user: userData,
          isAuthenticated: true,
          error: null,
          isLoading: false,
        });
        // Optionally, set token in apiClient default headers if not handled by interceptor on initial load
        // apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        console.log("User logged in, token and user data set.");
      },

      logout: () => {
        // Optionally call backend logout endpoint
        // apiClient.post('/auth/logout').catch(err => console.error("Logout API call failed", err));
        console.log("User logging out.");
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        });
        // delete apiClient.defaults.headers.common['Authorization'];
        if (typeof window !== 'undefined') {
            localStorage.removeItem('dummy-jwt-token'); // Placeholder cleanup
            localStorage.removeItem('dummy-refresh-token'); // Placeholder cleanup
        }
      },

      setToken: (token, refreshToken) => {
        set({ token, refreshToken, isAuthenticated: !!token });
        // if (token) {
        //   apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        // } else {
        //   delete apiClient.defaults.headers.common['Authorization'];
        // }
      },

      setUser: (userData) => {
        set({ user: userData });
      },

      clearAuth: () => {
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        });
        // delete apiClient.defaults.headers.common['Authorization'];
      },

      loadAuthFromStorage: () => {
        // This function is mostly illustrative if persist middleware is used,
        // as persist middleware hydrates the store automatically.
        // However, it can be useful for initial setup or specific re-hydration logic.
        const state = get();
        if (state.token && state.user) {
          set({ isAuthenticated: true });
          // apiClient.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
          console.log("Auth state loaded from storage.");
        }
      }
    }),
    {
      name: 'auth-storage', // name of the item in the storage (must be unique)
      storage: createJSONStorage(() => localStorage), // (optional) by default, 'localStorage' is used
      onRehydrateStorage: (state) => {
        console.log("Auth store rehydrated");
        // if (state?.token) {
        //    apiClient.defaults.headers.common['Authorization'] = `Bearer ${state.token}`;
        // }
        return (state, error) => {
          if (error) {
            console.error("Failed to rehydrate auth store:", error);
          }
        }
      }
    }
  )
);

// Call loadAuthFromStorage on store initialization if not relying solely on persist rehydration
// This is often not needed if persist middleware handles everything.
// useAuthStore.getState().loadAuthFromStorage();

export default useAuthStore;
