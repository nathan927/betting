"use client";

import React from 'react';
import { Card, Title, Text, BarList, DonutChart, Legend, Flex, Metric } from '@tremor/react';
import { DropletsIcon, ThermometerIcon, WindIcon, TrendingUpIcon } from 'lucide-react'; // Example icons

interface RaceAnalyticsData {
  trackCondition?: string; // e.g., "Good", "Soft", "Heavy"
  weather?: {
    temperature?: number; // Celsius
    humidity?: number; // Percentage
    windSpeed?: number; // km/h
    description?: string;
  };
  averageSpeed?: number; // km/h or m/s for the race type
  keyFactors?: string[]; // Factors influencing predictions (e.g., "Form", "Jockey", "Trainer", "Draw")
  paceAnalysis?: { // Example pace analysis
    earlyPaceHorses?: { name: string, rating: number }[];
    latePaceHorses?: { name: string, rating: number }[];
  };
  historicalWinRates?: { // Example historical data
    byDraw?: { draw: number, winRate: number }[];
    byJockey?: { jockey: string, winRate: number }[];
  }
}

interface RaceAnalyticsProps {
  analytics?: RaceAnalyticsData;
  raceName?: string;
}

const valueFormatter = (number: number) => `${Intl.NumberFormat('us').format(number).toString()}`;

export default function RaceAnalytics({ analytics, raceName }: RaceAnalyticsProps) {
  if (!analytics) {
    return (
      <Card>
        <Title>{raceName ? `Analytics for ${raceName}` : "Race Analytics"}</Title>
        <Text className="mt-2">No analytics data available for this race.</Text>
      </Card>
    );
  }

  const keyFactorsData = analytics.keyFactors?.map(factor => ({ name: factor, value: Math.random() * 100 })) || []; // Placeholder values
  const paceData = [
    { name: "Early Pace Leaders", value: analytics.paceAnalysis?.earlyPaceHorses?.length || 0 },
    { name: "Late Pace Surges", value: analytics.paceAnalysis?.latePaceHorses?.length || 0 }
  ];


  return (
    <Card>
      <Title>{raceName ? `Analytics for ${raceName}` : "Race Analytics"}</Title>

      <Grid numItemsSm={1} numItemsMd={2} className="gap-6 mt-4">
        <div>
          <Text className="font-medium">Race Conditions</Text>
          <Flex className="mt-2 items-start">
            <DropletsIcon className="h-5 w-5 text-tremor-content dark:text-dark-tremor-content mr-2" />
            <Text>Track: {analytics.trackCondition || "N/A"}</Text>
          </Flex>
          {analytics.weather && (
            <>
              <Flex className="mt-1 items-start">
                <ThermometerIcon className="h-5 w-5 text-tremor-content dark:text-dark-tremor-content mr-2" />
                <Text>Temp: {analytics.weather.temperature?.toFixed(1)}°C</Text>
              </Flex>
              <Flex className="mt-1 items-start">
                <WindIcon className="h-5 w-5 text-tremor-content dark:text-dark-tremor-content mr-2" />
                <Text>Wind: {analytics.weather.windSpeed} km/h</Text>
              </Flex>
              <Text className="text-xs mt-1">{analytics.weather.description}</Text>
            </>
          )}
        </div>

        <div>
          <Text className="font-medium">Expected Average Speed</Text>
          <Metric className="mt-1">{analytics.averageSpeed ? `${analytics.averageSpeed.toFixed(1)} km/h` : "N/A"}</Metric>
        </div>
      </Grid>

      {keyFactorsData.length > 0 && (
        <div className="mt-6">
          <Text className="font-medium">Key Predictive Factors</Text>
          <BarList data={keyFactorsData} className="mt-2" valueFormatter={valueFormatter} />
        </div>
      )}

      {analytics.paceAnalysis && (
        <div className="mt-6">
          <Text className="font-medium">Pace Analysis</Text>
           <DonutChart
            className="mt-2 h-40"
            data={paceData}
            category="value"
            index="name"
            valueFormatter={valueFormatter}
            colors={["cyan", "purple"]}
          />
          <Legend categories={["Early Pace Leaders", "Late Pace Surges"]} colors={["cyan", "purple"]} className="mt-2" />
        </div>
      )}
       {/* Add more analytics displays here, e.g., historical data */}
    </Card>
  );
}
