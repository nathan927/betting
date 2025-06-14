"use client";

import React from 'react';
import { AreaChart, Card, Title } from '@tremor/react';

interface PerformanceDataPoint {
  date: string; // e.g., "2024-07-01"
  profit: number;
  // Optional: cumulativeProfit?: number;
}

interface PerformanceChartProps {
  data: PerformanceDataPoint[];
  title?: string;
  height?: string;
}

const valueFormatter = (number: number) => `$${new Intl.NumberFormat('us').format(number).toString()}`;

export default function PerformanceChart({
  data,
  title = "Profit Over Time",
  height = "h-72"
}: PerformanceChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card>
        <Title>{title}</Title>
        <div className={`flex items-center justify-center ${height}`}>
          <p className="text-tremor-content dark:text-dark-tremor-content">No performance data available.</p>
        </div>
      </Card>
    );
  }

  // Optional: Calculate cumulative profit if not provided
  const chartData = data.reduce((acc, current, index) => {
    const cumulativeProfit = (index > 0 ? acc[index - 1].cumulativeProfit : 0) + current.profit;
    acc.push({
      ...current,
      cumulativeProfit,
    });
    return acc;
  }, [] as Array<PerformanceDataPoint & { cumulativeProfit: number }>);


  return (
    <Card>
      <Title>{title}</Title>
      <AreaChart
        className={height + " mt-4"}
        data={chartData}
        index="date"
        categories={['profit', 'cumulativeProfit']}
        colors={['blue', 'emerald']}
        valueFormatter={valueFormatter}
        yAxisWidth={60}
        showLegend={true}
        curveType="monotone" // or "linear"
      />
    </Card>
  );
}
