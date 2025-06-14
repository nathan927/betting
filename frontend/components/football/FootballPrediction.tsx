"use client";

import React from 'react';
import { Card, Title, Text, BarList, DonutChart, Legend, Flex, Metric, Grid, Col } from '@tremor/react';
import { ShieldCheckIcon, ShieldAlertIcon, ShieldIcon, UsersIcon, CalendarDaysIcon, BarChartIcon } from 'lucide-react'; // Example icons

interface MatchDetails {
  homeTeam: string;
  awayTeam: string;
  league: string;
  startTime: string; // ISO string
}

interface PredictionData {
  matchId: string;
  homeWinProbability: number;
  awayWinProbability: number;
  drawProbability: number;
  predictedScore?: string;
  keyFactors?: string[];
  confidence?: number;
  // Add more specific fields as needed, e.g., expected goals, player props, etc.
  insights?: {
    homeTeamForm?: string; // e.g., "WWLDW"
    awayTeamForm?: string;
    headToHeadSummary?: string; // e.g., "Home team won 3 of last 5"
  }
}

interface FootballPredictionProps {
  prediction?: PredictionData;
  matchDetails?: MatchDetails;
}

const probabilityValueFormatter = (number: number) => `${(number * 100).toFixed(1)}%`;

export default function FootballPrediction({ prediction, matchDetails }: FootballPredictionProps) {
  if (!prediction || !matchDetails) {
    return (
      <Card>
        <Title>Football Match Prediction</Title>
        <Text className="mt-2">No prediction data available for this match.</Text>
      </Card>
    );
  }

  const outcomeData = [
    { name: matchDetails.homeTeam, value: prediction.homeWinProbability },
    { name: "Draw", value: prediction.drawProbability },
    { name: matchDetails.awayTeam, value: prediction.awayWinProbability },
  ];

  const keyFactorsDisplay = prediction.keyFactors?.map(factor => ({ name: factor, value: Math.random() * 10 })) || []; // Placeholder values

  return (
    <Card>
      <Title>
        {matchDetails.homeTeam} vs {matchDetails.awayTeam}
      </Title>
      <Text>{matchDetails.league} - {new Date(matchDetails.startTime).toLocaleString()}</Text>

      <Grid numItemsSm={1} numItemsMd={2} className="gap-6 mt-6">
        <Col>
          <Text className="font-medium">Match Outcome Probabilities</Text>
          <DonutChart
            className="mt-2 h-52"
            data={outcomeData}
            category="value"
            index="name"
            valueFormatter={probabilityValueFormatter}
            colors={["indigo", "slate", "violet"]}
          />
          <Legend
            categories={[matchDetails.homeTeam, "Draw", matchDetails.awayTeam]}
            colors={["indigo", "slate", "violet"]}
            className="mt-3"
          />
        </Col>
        <Col>
          <Text className="font-medium">Prediction Details</Text>
          <Flex className="mt-2 items-center">
            <BarChartIcon className="h-5 w-5 text-tremor-content dark:text-dark-tremor-content mr-2" />
            <Text>Predicted Score: {prediction.predictedScore || "N/A"}</Text>
          </Flex>
          {prediction.confidence && (
            <Flex className="mt-1 items-center">
              <ShieldCheckIcon className="h-5 w-5 text-tremor-content dark:text-dark-tremor-content mr-2" />
              <Text>Confidence: {(prediction.confidence * 100).toFixed(1)}%</Text>
            </Flex>
          )}
          {prediction.insights?.homeTeamForm && (
             <Text className="text-xs mt-1">Home Form: {prediction.insights.homeTeamForm}</Text>
          )}
          {prediction.insights?.awayTeamForm && (
             <Text className="text-xs mt-1">Away Form: {prediction.insights.awayTeamForm}</Text>
          )}
           {prediction.insights?.headToHeadSummary && (
             <Text className="text-xs mt-1">H2H: {prediction.insights.headToHeadSummary}</Text>
          )}
        </Col>
      </Grid>

      {keyFactorsDisplay.length > 0 && (
        <div className="mt-6">
          <Text className="font-medium">Key Predictive Factors</Text>
          <BarList data={keyFactorsDisplay} className="mt-2" valueFormatter={(val) => `${val.toFixed(1)}/10 impact`} />
        </div>
      )}
    </Card>
  );
}
