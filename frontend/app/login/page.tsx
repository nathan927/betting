"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/stores/authStore'; // Assuming path
import { loginUser } from '@/services/api'; // Assuming path
import { Card, Title, TextInput, Button, Text, Alert } from '@tremor/react';
import { LogInIcon, AlertCircleIcon } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const { login: loginToStore } = useAuthStore();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // const response = await loginUser({ username, password });
      // Mock response for now as backend is not running
      const response = await new Promise<any>((resolve, reject) => {
        setTimeout(() => {
          if (username === "testuser" && password === "password") {
            resolve({
              accessToken: "fake-access-token",
              refreshToken: "fake-refresh-token",
              user: { id: "1", username: "testuser", email: "test@example.com", role: "user" }
            });
          } else {
            reject(new Error("Invalid credentials"));
          }
        }, 1000);
      });


      loginToStore(response.accessToken, response.refreshToken, response.user);
      router.push('/'); // Redirect to dashboard or intended page
    } catch (err: any) {
      setError(err.message || 'Failed to login. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 dark:bg-gray-900">
      <Card className="max-w-md w-full p-6">
        <Title className="text-center text-2xl mb-6">Login to Betting System</Title>
        {error && (
          <Alert title="Login Failed" color="rose" icon={AlertCircleIcon} className="mb-4">
            {error}
          </Alert>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Text className="mb-1">Username or Email</Text>
            <TextInput
              placeholder="yourname or your@email.com"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <div>
            <Text className="mb-1">Password</Text>
            <TextInput
              type="password"
              placeholder="Your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <Button
            type="submit"
            className="w-full"
            icon={LogInIcon}
            loading={isLoading}
            disabled={isLoading}
          >
            Login
          </Button>
        </form>
        <Text className="mt-4 text-center">
          Don&apos;t have an account?{' '}
          <a href="/register" className="text-tremor-brand hover:text-tremor-brand-emphasis">
            Register here
          </a>
        </Text>
      </Card>
    </div>
  );
}
