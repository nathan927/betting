"use client";

import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getRecentBets } from '@/services/api'; // Assuming this function exists in api.ts
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
  Text,
  Badge,
} from '@tremor/react';
import { format } from 'date-fns'; // For date formatting

interface Bet {
  id: string;
  eventName: string;
  marketType: string;
  selection: string;
  stake: number;
  odds: number;
  status: 'pending' | 'won' | 'lost' | 'void' | 'cashout';
  placedAt: string; // ISO string
  potentialPayout: number;
  actualPayout?: number;
}

// Mocked getRecentBets if not available in api.ts yet
// const getRecentBets = async (): Promise<Bet[]> => {
//   console.log("Fetching recent bets (mocked)");
//   return new Promise(resolve => setTimeout(() => resolve([
//     { id: '1', eventName: 'Man Utd vs Liverpool', marketType: '1X2', selection: 'Man Utd', stake: 100, odds: 2.5, status: 'won', placedAt: new Date().toISOString(), potentialPayout: 250, actualPayout: 250 },
//     { id: '2', eventName: 'Chelsea vs Arsenal', marketType: 'O/U 2.5', selection: 'Over 2.5', stake: 50, odds: 1.8, status: 'lost', placedAt: new Date(Date.now() - 86400000).toISOString(), potentialPayout: 90 },
//     { id: '3', eventName: 'Real Madrid vs Barcelona', marketType: 'AH', selection: 'Real Madrid -0.5', stake: 75, odds: 2.1, status: 'pending', placedAt: new Date(Date.now() - 172800000).toISOString(), potentialPayout: 157.5 },
//   ]), 1000));
// };


const statusColors: { [key in Bet['status']]: string } = {
  pending: 'yellow',
  won: 'emerald',
  lost: 'rose',
  void: 'slate',
  cashout: 'sky',
};

export default function RecentBetsTable() {
  const { data: bets, isLoading, error } = useQuery<Bet[], Error>({
    queryKey: ['recentBets'],
    queryFn: getRecentBets, // This should be an async function from api.ts
  });

  if (isLoading) return <Text>Loading recent bets...</Text>;
  if (error) return <Text>Error loading bets: {error.message}</Text>;
  if (!bets || bets.length === 0) return <Text>No recent bets found.</Text>;

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Event</TableHeaderCell>
          <TableHeaderCell>Selection</TableHeaderCell>
          <TableHeaderCell className="text-right">Stake</TableHeaderCell>
          <TableHeaderCell className="text-right">Odds</TableHeaderCell>
          <TableHeaderCell className="text-right">Payout</TableHeaderCell>
          <TableHeaderCell>Status</TableHeaderCell>
          <TableHeaderCell>Placed At</TableHeaderCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {bets.map((bet) => (
          <TableRow key={bet.id}>
            <TableCell>
              <Text>{bet.eventName}</Text>
              <Text className="text-xs text-gray-500">{bet.marketType}</Text>
            </TableCell>
            <TableCell>{bet.selection}</TableCell>
            <TableCell className="text-right">${bet.stake.toFixed(2)}</TableCell>
            <TableCell className="text-right">{bet.odds.toFixed(2)}</TableCell>
            <TableCell className="text-right">
              {bet.status === 'won' || bet.status === 'cashout'
                ? `$${(bet.actualPayout || 0).toFixed(2)}`
                : `$${bet.potentialPayout.toFixed(2)}`}
            </TableCell>
            <TableCell>
              <Badge color={statusColors[bet.status]} className="capitalize">
                {bet.status}
              </Badge>
            </TableCell>
            <TableCell>{format(new Date(bet.placedAt), 'MMM d, yyyy HH:mm')}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
