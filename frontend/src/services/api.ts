import axios from 'axios';

// Base URL defaults to /api/v1 so local Vite proxy directs requests to http://127.0.0.1:8000
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Standard error logging without crashing application
    if (error.response) {
      console.warn(`[API Response Error] ${error.response.status} from ${error.config?.url}`);
    } else if (error.request) {
      console.warn(`[API Network Error] No response received from ${error.config?.url}`);
    }
    return Promise.reject(error);
  }
);
