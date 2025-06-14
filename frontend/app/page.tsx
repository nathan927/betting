"use client";

import { Card, Metric, Text, Flex, Grid, Title, Subtitle, DonutChart, BarChart, AreaChart } from "@tremor/react";
import { DollarSignIcon, UsersIcon, TrendingUpIcon, ListChecksIcon } from "lucide-react";
import LiveOddsPanel from "@/components/betting/LiveOddsPanel";
import RecentBetsTable from "@/components/betting/RecentBetsTable";
import PerformanceChart from "@/components/charts/PerformanceChart"; // Assuming this was created/detailed in markdown
// import { useAuthStore } from "@/stores/authStore"; // Assuming this was created/detailed
// import { useDashboardStats } from "@/hooks/useDashboardData"; // Assuming custom hook for data

// Mock data for demonstration if hooks/API are not fully implemented yet
const mockUser = { username: "ProBettor123" };
const mockStats = {
  totalBalance: 12580.50,
  activeUsers: 1345,
  dailyProfit: 750.20,
  pendingBets: 15,
  profitBySport: [
    { name: 'Football', value: 12000 },
    { name: 'Horse Racing', value: 8500 },
    { name: 'Basketball', value: 6200 },
    { name: 'Tennis', value: 4000 },
  ],
  performanceHistory: [
    { date: '2024-07-01', Profit: 400 },
    { date: '2024-07-02', Profit: 650 },
    { date: '2024-07-03', Profit: 300 },
    { date: '2024-07-04', Profit: 800 },
    { date: '2024-07-05', Profit: 550 },
  ]
};

export default function DashboardPage() {
  // const { user } = useAuthStore(); // Real implementation
  // const { data: stats, isLoading, error } = useDashboardStats(); // Real implementation
  const user = mockUser; // Using mock
  const stats = mockStats; // Using mock
  const isLoading = false; // Mock
  const error = null; // Mock

  if (isLoading) return <Text>Loading dashboard data...</Text>;
  if (error) return <Text>Error loading dashboard: {/* @ts-ignore */}{error.message}</Text>;

  const dataFormatter = (number: number) => `$${Intl.NumberFormat('us').format(number).toString()}`;

  return (
    <div className="space-y-6 lg:space-y-8">
      <Title>Welcome, {user?.username || "Guest"}!</Title>
      <Subtitle>Your central hub for betting activity and insights.</Subtitle>

      <Grid numItemsMd={2} numItemsLg={4} className="gap-6">
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Flex alignItems="center" className="space-x-3">
            <DollarSignIcon className="h-6 w-6 text-tremor-brand" />
            <Text>Total Balance</Text>
          </Flex>
          <Metric className="mt-1">${stats?.totalBalance.toFixed(2) || "0.00"}</Metric>
        </Card>
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Flex alignItems="center" className="space-x-3">
            <UsersIcon className="h-6 w-6 text-tremor-brand" />
            <Text>Active Users Online</Text>
          </Flex>
          <Metric className="mt-1">{stats?.activeUsers || 0}</Metric>
        </Card>
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Flex alignItems="center" className="space-x-3">
            <TrendingUpIcon className="h-6 w-6 text-emerald-500" />
            <Text>Today&apos;s Profit</Text>
          </Flex>
          <Metric className={`mt-1 ${stats?.dailyProfit >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
            ${stats?.dailyProfit.toFixed(2) || "0.00"}
          </Metric>
        </Card>
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Flex alignItems="center" className="space-x-3">
            <ListChecksIcon className="h-6 w-6 text-tremor-brand" />
            <Text>Pending Bets</Text>
          </Flex>
          <Metric className="mt-1">{stats?.pendingBets || 0}</Metric>
        </Card>
      </Grid>

      <Grid numItemsMd={1} numItemsLg={3} className="gap-6 mt-6">
        <Card className="lg:col-span-2 hover:shadow-lg transition-shadow duration-300">
          <Title>Betting Performance Overview</Title>
          {/* Using PerformanceChart which might use AreaChart internally */}
          <PerformanceChart data={stats?.performanceHistory || []} height="h-80" />
        </Card>
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Title>Profit by Sport</Title>
          <DonutChart
            className="mt-6 h-80"
            data={stats?.profitBySport || []}
            category="value"
            index="name"
            valueFormatter={dataFormatter}
            colors={["indigo", "cyan", "amber", "rose", "emerald", "slate"]}
          />
        </Card>
      </Grid>

      <Grid numItemsMd={1} numItemsLg={2} className="gap-6 mt-6">
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Title>Live Odds Feed</Title>
          <Text className="mt-1">Real-time odds from various bookmakers.</Text>
          <div className="mt-4 max-h-96 overflow-y-auto"> {/* Added scroll for long content */}
            <LiveOddsPanel />
          </div>
        </Card>
        <Card className="hover:shadow-lg transition-shadow duration-300">
          <Title>Recent Bets</Title>
          <Text className="mt-1">A log of your most recent betting activity.</Text>
          <div className="mt-4 max-h-96 overflow-y-auto"> {/* Added scroll for long content */}
            <RecentBetsTable />
          </div>
        </Card>
      </Grid>
    </div>
  );
}
