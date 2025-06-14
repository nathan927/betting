"use client";

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getHorseRacingRaces,
  getHorseRacingPrediction,
  // Other potential API calls: getRaceDetails, getHorseForm, getJockeyStats, etc.
} from '@/services/api'; // Assuming path

// Typings (should match actual API response structures)
export interface Race {
  id: string;
  name: string;
  venue: string;
  raceNumber: number;
  startTime: string; // ISO string
  trackCondition?: string;
  distance?: number; // meters
}

export interface HorsePrediction {
  horseNumber: number;
  horseName: string;
  winProbability: number;
  predictedPosition: number;
  odds?: number;
  form?: string;
  jockey?: string;
  trainer?: string;
}

export interface RacePredictionResponse {
  raceId: string;
  predictions: HorsePrediction[];
  analytics?: any; // Define more specific type if available
}

// Hook to fetch list of races for a given date
export const useRaces = (date?: string) => {
  return useQuery<Race[], Error>({
    queryKey: ['horseRaces', date],
    queryFn: () => getHorseRacingRaces(date),
    staleTime: 1000 * 60 * 15, // 15 minutes
    enabled: !!date, // Only fetch if date is provided
  });
};

// Hook to fetch prediction for a specific race
export const useRacePrediction = (raceId: string | null) => {
  return useQuery<RacePredictionResponse, Error>({
    queryKey: ['horseRacePrediction', raceId],
    queryFn: () => {
      if (!raceId) return Promise.reject(new Error("Race ID is required"));
      return getHorseRacingPrediction(raceId);
    },
    enabled: !!raceId, // Only fetch if raceId is provided
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};

// Example of a mutation hook if there were actions like placing a "simulated bet" or "saving prediction preferences"
// export const useUpdateRaceWatchlist = () => {
//   const queryClient = useQueryClient();
//   return useMutation({
//     mutationFn: async (raceId: string) => {
//       // Replace with actual API call, e.g., apiClient.post(`/watchlist/horse-racing/${raceId}`)
//       console.log(`Toggling watchlist status for race ${raceId} (simulated)`);
//       return { success: true, raceId };
//     },
//     onSuccess: (data) => {
//       // Invalidate and refetch relevant queries after mutation
//       queryClient.invalidateQueries({ queryKey: ['userWatchlist'] });
//       console.log(`Watchlist updated for race ${data.raceId}`);
//     },
//     onError: (error) => {
//       console.error("Failed to update watchlist:", error);
//     }
//   });
// };

// Placeholder for other potential hooks related to horse racing
// e.g., useHorseDetails, useJockeyPerformance, useTrainerPerformance

export default { useRaces, useRacePrediction };
