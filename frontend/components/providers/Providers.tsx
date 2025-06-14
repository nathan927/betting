"use client";

import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
// import { AuthProvider } from "@/stores/authStore"; // If authStore provides a context provider

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1, // Retry failed requests once
    },
  },
});

export default function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  return (
    <QueryClientProvider client={queryClient}>
      {/* If AuthProvider is a context provider, wrap it here */}
      {/* <AuthProvider> */}
      {mounted ? children : null} {/* Ensure children are only rendered on the client for some providers */}
      {/* </AuthProvider> */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
