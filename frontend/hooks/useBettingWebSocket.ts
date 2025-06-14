"use client";

import { useEffect, useState, useRef, useCallback } from 'react';
import _useWebSocket, { ReadyState } from 'react-use-websocket'; // Renamed to avoid conflict
// import { useAuthStore } from '@/stores/authStore'; // Actual path to Zustand store

interface WebSocketOptions {
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void; // For raw message handling if needed
  shouldReconnect?: (event: CloseEvent) => boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  share?: boolean; // Share WebSocket instance
  filter?: (message: MessageEvent) => boolean; // Filter messages before parsing
}

const DEFAULT_RECONNECT_INTERVAL = 3000; // Start with 3 seconds
const MAX_RECONNECT_INTERVAL = 30000; // Max 30 seconds
const DEFAULT_RECONNECT_ATTEMPTS = Infinity; // Reconnect indefinitely by default

export default function useBettingWebSocket<T = any>(
  getSocketUrl: () => string | null, // Function to dynamically get URL (can include token)
  options: WebSocketOptions = {}
) {
  const [socketUrl, setSocketUrl] = useState<string | null>(null);
  // const token = useAuthStore(state => state.token); // Example: get token from Zustand store
  const token = typeof window !== 'undefined' ? localStorage.getItem("dummy-jwt-token") : null; // Placeholder

  // Effect to update socket URL when token or generator changes
  useEffect(() => {
    const newUrl = getSocketUrl();
    if (newUrl) {
      // Append token if not already in URL from getSocketUrl function
      const urlWithToken = new URL(newUrl);
      if (token && !urlWithToken.searchParams.has('token')) {
        urlWithToken.searchParams.append('token', token);
      }
      setSocketUrl(urlWithToken.toString());
    } else {
      setSocketUrl(null);
    }
  }, [getSocketUrl, token]);

  const reconnectAttemptsRef = useRef(0);

  const {
    sendMessage: sendRawMessage,
    lastMessage: rawLastMessage,
    readyState,
    getWebSocket,
  } = _useWebSocket(socketUrl, {
    onOpen: (event) => {
      console.log(`WebSocket connected to ${socketUrl}`);
      reconnectAttemptsRef.current = 0;
      if (options.onOpen) options.onOpen(event);
    },
    onClose: (event) => {
      console.log(`WebSocket disconnected from ${socketUrl}`);
      if (options.onClose) options.onClose(event);
    },
    onError: (event) => {
      console.error('WebSocket error:', event);
      if (options.onError) options.onError(event);
    },
    onMessage: options.onMessage,
    shouldReconnect: (closeEvent) => {
      if (reconnectAttemptsRef.current >= (options.reconnectAttempts ?? DEFAULT_RECONNECT_ATTEMPTS)) {
        console.log('Max reconnect attempts reached.');
        return false;
      }
      return options.shouldReconnect ? options.shouldReconnect(closeEvent) : true;
    },
    reconnectInterval: () => {
      const attempt = reconnectAttemptsRef.current;
      // Exponential backoff with jitter
      const jitter = Math.random() * 1000;
      const interval = Math.min(
        DEFAULT_RECONNECT_INTERVAL * Math.pow(2, attempt) + jitter,
        MAX_RECONNECT_INTERVAL
      );
      reconnectAttemptsRef.current = attempt + 1;
      console.log(`WebSocket attempting to reconnect (attempt ${attempt + 1}) in ${(interval / 1000).toFixed(1)}s`);
      return interval;
    },
    filter: options.filter || (() => true), // Process all messages by default for JSON parsing
    share: options.share ?? true, // Share by default
    retryOnError: true, // Automatically retry on connection errors
  }, socketUrl !== null); // Only connect if socketUrl is not null

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  const [parsedLastJsonMessage, setParsedLastJsonMessage] = useState<T | null>(null);

  useEffect(() => {
    if (rawLastMessage?.data) {
      try {
        setParsedLastJsonMessage(JSON.parse(rawLastMessage.data as string) as T);
      } catch (e) {
        console.error("Failed to parse WebSocket JSON message:", e, rawLastMessage.data);
        setParsedLastJsonMessage(null); // Or handle as non-JSON message
      }
    }
  }, [rawLastMessage]);

  const sendMessage = useCallback((message: object | string | ArrayBuffer | Blob | ArrayBufferView) => {
    if (readyState === ReadyState.OPEN) {
      const dataToSend = typeof message === 'object' ? JSON.stringify(message) : message;
      sendRawMessage(dataToSend);
    } else {
      console.warn("Cannot send message, WebSocket is not open. Current state:", connectionStatus);
    }
  }, [readyState, sendRawMessage, connectionStatus]);

  return {
    sendMessage,
    lastJsonMessage: parsedLastJsonMessage, // Parsed JSON message
    lastMessage: rawLastMessage, // Raw message event
    readyState,
    connectionStatus,
    getWebSocket, // For advanced use cases, e.g., checking bufferedAmount
  };
}
