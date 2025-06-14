import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css"; // Should exist or be created with defaults
import Providers from "@/components/providers/Providers";
import Navigation from "@/components/navigation/Navigation";
import { Toaster } from "react-hot-toast"; // Assuming react-hot-toast for notifications

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: {
    default: "Advanced Betting Platform",
    template: "%s | Betting Platform",
  },
  description: "Next-generation platform for sports betting, live odds, and advanced analytics.",
  keywords: ["betting", "sports", "live odds", "analytics", "horse racing", "football"],
  authors: [{ name: "Betting System Inc." }],
  robots: {
    index: true,
    follow: true,
  },
  // openGraph: { ... } // Add OpenGraph metadata
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "black" },
  ],
  initialScale: 1,
  width: 'device-width',
  // maximumScale: 1, // Consider accessibility
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <body className="font-inter antialiased bg-tremor-background text-tremor-content dark:bg-dark-tremor-background dark:text-dark-tremor-content">
        <Providers>
          <div className="flex flex-col min-h-screen">
            <Navigation />
            <main className="flex-grow container mx-auto px-4 py-8 sm:px-6 lg:px-8">
              {children}
            </main>
            <footer className="text-center p-4 sm:p-6 border-t border-tremor-border dark:border-dark-tremor-border">
              <p className="text-sm text-tremor-content-subtle dark:text-dark-tremor-content-subtle">
                &copy; {new Date().getFullYear()} Betting System Inc. All rights reserved. Please bet responsibly.
              </p>
            </footer>
          </div>
          <Toaster position="top-right" reverseOrder={false} />
        </Providers>
      </body>
    </html>
  );
}
