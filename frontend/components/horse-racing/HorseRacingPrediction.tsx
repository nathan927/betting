"use client";

import React from 'react';
import { Card, Title, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell, Text, Badge } from '@tremor/react';
import { TrendingUpIcon, TrendingDownIcon, AwardIcon } from 'lucide-react';

interface Prediction {
  horseNumber: number;
  horseName: string;
  winProbability: number;
  predictedPosition: number;
  odds?: number; // Current odds from bookmaker
  form?: string; // e.g., "1-2-3"
  jockey?: string;
  trainer?: string;
}

interface HorseRacingPredictionProps {
  predictions?: Prediction[];
  raceName?: string;
}

const probabilityColor = (prob: number): "emerald" | "yellow" | "rose" => {
  if (prob > 0.3) return "emerald";
  if (prob > 0.15) return "yellow";
  return "rose";
};

export default function HorseRacingPrediction({ predictions, raceName }: HorseRacingPredictionProps) {
  if (!predictions || predictions.length === 0) {
    return (
      <Card>
        <Title>{raceName ? `Predictions for ${raceName}` : "Horse Racing Predictions"}</Title>
        <Text className="mt-2">No prediction data available for this race.</Text>
      </Card>
    );
  }

  // Sort by predicted position, then by win probability
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (a.predictedPosition !== b.predictedPosition) {
      return a.predictedPosition - b.predictedPosition;
    }
    return b.winProbability - a.winProbability;
  });

  return (
    <Card>
      <Title>{raceName ? `Predictions for ${raceName}` : "Horse Racing Predictions"}</Title>
      <Table className="mt-4">
        <TableHead>
          <TableRow>
            <TableHeaderCell>Pos.</TableHeaderCell>
            <TableHeaderCell>Horse</TableHeaderCell>
            <TableHeaderCell className="text-right">Win Prob.</TableHeaderCell>
            <TableHeaderCell className="text-right">Odds</TableHeaderCell>
            <TableHeaderCell>Jockey / Trainer</TableHeaderCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedPredictions.map((pred) => (
            <TableRow key={pred.horseNumber} className={pred.predictedPosition === 1 ? "bg-tremor-background-subtle dark:bg-dark-tremor-background-subtle" : ""}>
              <TableCell>
                <Badge color={pred.predictedPosition <=3 ? "amber" : "slate"} icon={pred.predictedPosition === 1 ? AwardIcon : undefined}>
                    {pred.predictedPosition}
                </Badge>
              </TableCell>
              <TableCell>
                <Text className="font-medium">{pred.horseNumber}. {pred.horseName}</Text>
                {pred.form && <Text className="text-xs">Form: {pred.form}</Text>}
              </TableCell>
              <TableCell className="text-right">
                <Badge color={probabilityColor(pred.winProbability)}>
                  {(pred.winProbability * 100).toFixed(1)}%
                </Badge>
              </TableCell>
              <TableCell className="text-right">
                {pred.odds ? pred.odds.toFixed(2) : <Text>-</Text>}
              </TableCell>
              <TableCell>
                {pred.jockey && <Text className="text-xs">{pred.jockey}</Text>}
                {pred.trainer && <Text className="text-xs text-gray-500">{pred.trainer}</Text>}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
}
