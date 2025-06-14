"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
// import { useAuthStore } from '@/stores/authStore'; // Not typically used directly on register, but could be for auto-login
import { registerUser } from '@/services/api'; // Assuming path
import { Card, Title, TextInput, Button, Text, Alert } from '@tremor/react';
import { UserPlusIcon, AlertCircleIcon } from 'lucide-react';

export default function RegisterPage() {
  const router = useRouter();
  // const { login: loginToStore } = useAuthStore(); // If auto-login after register
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }

    setIsLoading(true);

    try {
      // const response = await registerUser({ username, email, password });
      // Mock response for now
      const response = await new Promise<any>((resolve, reject) => {
        setTimeout(() => {
          if (email === "existing@example.com") {
            reject(new Error("User with this email already exists."));
          } else {
            resolve({
              message: "Registration successful! Please login.",
              user: { id: "2", username, email, role: "user" }
            });
          }
        }, 1000);
      });

      setSuccessMessage(response.message || "Registration successful! You can now log in.");
      // Optionally, automatically log in the user:
      // loginToStore(response.accessToken, response.refreshToken, response.user);
      // router.push('/');

      // Or redirect to login page after a delay
      setTimeout(() => {
        router.push('/login');
      }, 3000);

    } catch (err: any) {
      setError(err.message || 'Failed to register. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 dark:bg-gray-900">
      <Card className="max-w-md w-full p-6">
        <Title className="text-center text-2xl mb-6">Create your Account</Title>
        {error && (
          <Alert title="Registration Failed" color="rose" icon={AlertCircleIcon} className="mb-4">
            {error}
          </Alert>
        )}
        {successMessage && (
          <Alert title="Registration Successful" color="emerald" className="mb-4">
            {successMessage}
          </Alert>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Text className="mb-1">Username</Text>
            <TextInput
              placeholder="Choose a username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <div>
            <Text className="mb-1">Email</Text>
            <TextInput
              type="email"
              placeholder="your@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <div>
            <Text className="mb-1">Password</Text>
            <TextInput
              type="password"
              placeholder="Create a password (min. 8 characters)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <div>
            <Text className="mb-1">Confirm Password</Text>
            <TextInput
              type="password"
              placeholder="Confirm your password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              disabled={isLoading}
            />
          </div>
          <Button
            type="submit"
            className="w-full"
            icon={UserPlusIcon}
            loading={isLoading}
            disabled={isLoading}
          >
            Register
          </Button>
        </form>
        <Text className="mt-4 text-center">
          Already have an account?{' '}
          <a href="/login" className="text-tremor-brand hover:text-tremor-brand-emphasis">
            Login here
          </a>
        </Text>
      </Card>
    </div>
  );
}
