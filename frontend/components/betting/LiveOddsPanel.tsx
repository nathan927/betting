"use client";

import React, { useState, useEffect, useMemo } from 'react';
import useBettingWebSocket from '@/hooks/useBettingWebSocket';
import { Card, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell, Badge, Text, TextInput, MultiSelect, MultiSelectItem, Select, SelectItem } from '@tremor/react';
import { ArrowUpIcon, ArrowDownIcon, MinusIcon, SearchIcon, FilterIcon } from 'lucide-react';
import { sportKeys, marketKeys } from '@/lib/bettingData'; // Assuming these exist for filtering

// Define the structure of an odd update
interface OddUpdate {
  eventId: string;
  eventName?: string; // Optional: event name might come in initial load or separate feed
  marketId: string;
  marketName?: string; // Optional: market name might come in initial load
  selectionId: string;
  selectionName: string;
  price: number;
  bookmaker: string;
  timestamp: number; // Unix timestamp
}

// Internal state structure for display
interface DisplayOdd {
  selectionId: string;
  selectionName: string;
  price: number;
  previousPrice?: number;
  bookmaker: string;
  lastUpdate: number;
  movement: 'up' | 'down' | 'none';
}

interface DisplayMarket {
  marketId: string;
  marketName: string;
  odds: DisplayOdd[];
}

interface DisplayEvent {
  eventId: string;
  eventName: string;
  markets: DisplayMarket[];
}

const WS_URL = process.env.NEXT_PUBLIC_ODDS_WS_URL || 'ws://localhost:8080/ws/live-odds';

export default function LiveOddsPanel({ initialEventId }: { initialEventId?: string }) {
  const { lastJsonMessage, connectionStatus } = useBettingWebSocket<OddUpdate>(WS_URL);
  const [events, setEvents] = useState<Map<string, DisplayEvent>>(new Map());
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSports, setSelectedSports] = useState<string[]>([]); // e.g. ['football', 'tennis']
  const [selectedMarkets, setSelectedMarkets] = useState<string[]>([]); // e.g. ['1X2', 'OU']

  useEffect(() => {
    if (lastJsonMessage) {
      const update = lastJsonMessage;
      // console.log("Received odd update:", update);

      setEvents(prevEvents => {
        const newEvents = new Map(prevEvents);
        let event = newEvents.get(update.eventId);

        if (!event) {
          event = {
            eventId: update.eventId,
            eventName: update.eventName || `Event ${update.eventId}`, // Fallback name
            markets: [],
          };
        } else {
          event = { ...event, eventName: update.eventName || event.eventName }; // Update name if provided
        }

        let market = event.markets.find(m => m.marketId === update.marketId);
        if (!market) {
          market = {
            marketId: update.marketId,
            marketName: update.marketName || `Market ${update.marketId}`, // Fallback name
            odds: [],
          };
          event.markets.push(market);
        } else {
           market = { ...market, marketName: update.marketName || market.marketName }; // Update name
           event.markets = event.markets.map(m => m.marketId === market!.marketId ? market! : m);
        }

        const existingOddIndex = market.odds.findIndex(o => o.selectionId === update.selectionId && o.bookmaker === update.bookmaker);
        let newOddData: DisplayOdd;

        if (existingOddIndex !== -1) {
          const oldOdd = market.odds[existingOddIndex];
          newOddData = {
            ...oldOdd,
            previousPrice: oldOdd.price,
            price: update.price,
            lastUpdate: update.timestamp,
            movement: update.price > oldOdd.price ? 'up' : (update.price < oldOdd.price ? 'down' : 'none'),
          };
          market.odds[existingOddIndex] = newOddData;
        } else {
          newOddData = {
            selectionId: update.selectionId,
            selectionName: update.selectionName,
            price: update.price,
            bookmaker: update.bookmaker,
            lastUpdate: update.timestamp,
            movement: 'none',
          };
          market.odds.push(newOddData);
        }

        // Sort odds within market, e.g., by selection name or a predefined order
        market.odds.sort((a, b) => a.selectionName.localeCompare(b.selectionName));
        newEvents.set(event.eventId, event);
        return newEvents;
      });
    }
  }, [lastJsonMessage]);

  const filteredEvents = useMemo(() => {
    return Array.from(events.values()).filter(event => {
      const nameMatch = event.eventName.toLowerCase().includes(searchTerm.toLowerCase());
      // Basic sport/market filtering (assuming eventName or marketName contains this info or it comes from API)
      const sportMatch = selectedSports.length === 0 || selectedSports.some(s => event.eventName.toLowerCase().includes(s));
      const marketMatch = selectedMarkets.length === 0 || event.markets.some(m => selectedMarkets.some(sm => m.marketName.toUpperCase().includes(sm)));

      return nameMatch && sportMatch && marketMatch && (initialEventId ? event.eventId === initialEventId : true);
    });
  }, [events, searchTerm, selectedSports, selectedMarkets, initialEventId]);

  const renderMovementIcon = (movement: DisplayOdd['movement']) => {
    if (movement === 'up') return <ArrowUpIcon className="h-4 w-4 text-emerald-500" />;
    if (movement === 'down') return <ArrowDownIcon className="h-4 w-4 text-rose-500" />;
    return <MinusIcon className="h-4 w-4 text-gray-500" />;
  };

  if (connectionStatus !== 'Open') {
    return <Text>Live odds feed: {connectionStatus}. Please wait...</Text>;
  }

  return (
    <Card className="h-full flex flex-col">
      <Flex className="space-x-2 mb-4" justifyContent="start" alignItems="center">
        <SearchIcon className="h-5 w-5 text-tremor-content-subtle dark:text-dark-tremor-content-subtle" />
        <TextInput
          placeholder="Search events..."
          value={searchTerm}
          onValueChange={setSearchTerm}
          className="flex-grow"
        />
        {/* Add sport/market filters if needed */}
        {/* <MultiSelect value={selectedSports} onValueChange={setSelectedSports} placeholder="Filter Sports..." className="max-w-xs">
          {sportKeys.map(sport => <MultiSelectItem key={sport.key} value={sport.key}>{sport.name}</MultiSelectItem>)}
        </MultiSelect>
        <MultiSelect value={selectedMarkets} onValueChange={setSelectedMarkets} placeholder="Filter Markets..." className="max-w-xs">
          {marketKeys.map(market => <MultiSelectItem key={market.key} value={market.key}>{market.name}</MultiSelectItem>)}
        </MultiSelect> */}
      </Flex>

      {filteredEvents.length === 0 && <Text>No live odds matching your criteria. Waiting for data...</Text>}

      <div className="flex-grow overflow-y-auto space-y-4">
        {filteredEvents.map(event => (
          <div key={event.eventId} className="p-2 border rounded-tremor-default dark:border-dark-tremor-border">
            <Text className="text-lg font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">{event.eventName}</Text>
            {event.markets.map(market => (
              <div key={market.marketId} className="mt-2">
                <Text className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">{market.marketName}</Text>
                <Table className="mt-1">
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Selection</TableHeaderCell>
                      <TableHeaderCell>Bookmaker</TableHeaderCell>
                      <TableHeaderCell className="text-right">Odds</TableHeaderCell>
                      <TableHeaderCell className="text-center">Trend</TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {market.odds.map(odd => (
                      <TableRow key={`${odd.selectionId}-${odd.bookmaker}`}>
                        <TableCell>{odd.selectionName}</TableCell>
                        <TableCell><Badge color="slate">{odd.bookmaker}</Badge></TableCell>
                        <TableCell className="text-right">
                           <span className={`font-mono px-2 py-1 rounded text-sm ${
                             odd.movement === 'up' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-800 dark:text-emerald-300' :
                             odd.movement === 'down' ? 'bg-rose-100 text-rose-700 dark:bg-rose-800 dark:text-rose-300' :
                             'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                           }`}>
                            {odd.price.toFixed(2)}
                           </span>
                        </TableCell>
                        <TableCell className="flex justify-center items-center">
                          {renderMovementIcon(odd.movement)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ))}
          </div>
        ))}
      </div>
    </Card>
  );
}

// Dummy data for sportKeys and marketKeys if not available from actual bettingData lib
// const sportKeys = [ {key: "football", name: "Football"}, {key: "tennis", name: "Tennis"}];
// const marketKeys = [ {key: "1X2", name: "Match Result (1X2)"}, {key: "OU", name: "Over/Under"}];

// Define lib/bettingData.ts if it doesn't exist
// export const sportKeys = [ {key: "football", name: "Football"}, {key: "tennis", name: "Tennis"}];
// export const marketKeys = [ {key: "1X2", name: "Match Result (1X2)"}, {key: "OU", name: "Over/Under"}];
