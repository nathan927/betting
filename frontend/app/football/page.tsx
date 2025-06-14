"use client";

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getFootballMatches, getFootballPrediction } from '@/services/api'; // Assuming path
import { Card, Title, Text, Select, SelectItem, Grid } from '@tremor/react';
import FootballPrediction from '@/components/football/FootballPrediction'; // Assuming path

interface Match {
  id: string;
  homeTeam: string;
  awayTeam: string;
  league: string;
  startTime: string; // ISO string
}

// Mocked API functions if not fully implemented in api.ts
// const getFootballMatches = async (league?: string, date?: string): Promise<Match[]> => {
//   console.log("Fetching football matches (mocked)", { league, date });
//   return new Promise(resolve => setTimeout(() => resolve([
//     { id: 'match1', homeTeam: 'Man City', awayTeam: 'Liverpool', league: 'Premier League', startTime: new Date().toISOString() },
//     { id: 'match2', homeTeam: 'Real Madrid', awayTeam: 'Barcelona', league: 'La Liga', startTime: new Date().toISOString() },
//   ]), 500));
// };

// const getFootballPrediction = async (matchId: string): Promise<any> => {
//   console.log("Fetching predictions for match (mocked):", matchId);
//   return new Promise(resolve => setTimeout(() => resolve({
//     matchId,
//     homeWinProbability: 0.45,
//     awayWinProbability: 0.25,
//     drawProbability: 0.30,
//     predictedScore: "2-1",
//     keyFactors: ["Team Form", "Head-to-Head", "Player Availability"]
//   }), 700));
// };

export default function FootballPage() {
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null);
  const [selectedLeague, setSelectedLeague] = useState<string | null>(null);
  // Further filters like date could be added here

  // Dummy list of leagues for selection
  const leagues = [
    { value: "premier-league", label: "Premier League (ENG)" },
    { value: "la-liga", label: "La Liga (ESP)" },
    { value: "bundesliga", label: "Bundesliga (GER)" },
    { value: "serie-a", label: "Serie A (ITA)" },
    { value: "ligue-1", label: "Ligue 1 (FRA)" },
  ];

  const { data: matches, isLoading: isLoadingMatches, error: matchesError } = useQuery<Match[], Error>({
    queryKey: ['footballMatches', selectedLeague], // Add date to queryKey if date filter is used
    queryFn: () => getFootballMatches(selectedLeague || undefined), // Pass undefined if no league selected
    // enabled: !!selectedLeague, // Or fetch all matches if no league is selected initially
  });

  const { data: predictionData, isLoading: isLoadingPrediction, error: predictionError } = useQuery<any, Error>({
    queryKey: ['footballPrediction', selectedMatchId],
    queryFn: () => getFootballPrediction(selectedMatchId!),
    enabled: !!selectedMatchId,
  });

  const matchOptions = useMemo(() => {
    if (!matches) return [];
    return matches.map(match => ({
      value: match.id,
      label: `${match.homeTeam} vs ${match.awayTeam} (${match.league} - ${new Date(match.startTime).toLocaleTimeString()})`,
    }));
  }, [matches]);

  return (
    <div className="space-y-6">
      <Title>Football Analytics</Title>
      <Text>Select a league and match to view detailed predictions.</Text>

      <Grid numItemsSm={1} numItemsMd={2} className="gap-6">
        <Card>
          <Text>Select League</Text>
          <Select
            value={selectedLeague || ""}
            onValueChange={(value) => {
              setSelectedLeague(value);
              setSelectedMatchId(null); // Reset match selection
            }}
            placeholder="All Leagues"
            className="mt-1"
          >
            <SelectItem value="">All Leagues</SelectItem>
            {leagues.map(league => (
              <SelectItem key={league.value} value={league.value}>
                {league.label}
              </SelectItem>
            ))}
          </Select>
        </Card>
        <Card>
          <Text>Select Match</Text>
          {isLoadingMatches ? <Text>Loading matches...</Text> : matchesError ? <Text>Error: {matchesError.message}</Text> : (
            <Select
              value={selectedMatchId || ""}
              onValueChange={setSelectedMatchId}
              placeholder="Select a match..."
              className="mt-1"
              disabled={!matches || matches.length === 0}
            >
              {matchOptions.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </Select>
          )}
          {matches && matches.length === 0 && !isLoadingMatches && <Text className="mt-1">No matches found for selected league/filters.</Text>}
        </Card>
      </Grid>

      {selectedMatchId && (
        <div className="mt-6">
          {isLoadingPrediction && <Text>Loading prediction data...</Text>}
          {predictionError && <Text>Error loading prediction: {predictionError.message}</Text>}
          {predictionData && (
            <FootballPrediction
              prediction={predictionData}
              matchDetails={matches?.find(m => m.id === selectedMatchId)}
            />
          )}
        </div>
      )}
    </div>
  );
}
