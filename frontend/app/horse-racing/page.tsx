"use client";

import React, { useState, useMemo, Suspense } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { Card, Title, Text, Select, SelectItem, Grid, Col, DateRangePicker, DateRangePickerValue } from '@tremor/react';
import { subDays, formatISO } from 'date-fns';

import { useRaces, useRacePrediction } from '@/hooks/useHorseRacingData'; // Assuming path
import HorseRacingPrediction from '@/components/horse-racing/HorseRacingPrediction'; // Assuming path
import RaceAnalytics from '@/components/horse-racing/RaceAnalytics'; // Assuming path
import { FilterIcon, CalendarIcon } from 'lucide-react';

// List of common venues for selection (can be fetched from API in a real app)
const venues = [
  { value: "HV", label: "Happy Valley (HK)" },
  { value: "ST", label: "Sha Tin (HK)" },
  { value: "Epsom", label: "Epsom Downs (UK)" },
  { value: "Ascot", label: "Ascot (UK)" },
  { value: "CD", label: "Churchill Downs (US)" },
  // Add more venues
];

function HorseRacingContent() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [selectedVenue, setSelectedVenue] = useState<string | undefined>(searchParams.get('venue') || undefined);
  const [dateRange, setDateRange] = useState<DateRangePickerValue>({
    from: searchParams.get('from') ? new Date(searchParams.get('from')!) : subDays(new Date(), 7),
    to: searchParams.get('to') ? new Date(searchParams.get('to')!) : new Date(),
  });
  const [selectedRaceId, setSelectedRaceId] = useState<string | undefined>(searchParams.get('raceId') || undefined);

  // Update URL when filters change
  const updateQueryParams = (newParams: Record<string, string | undefined>) => {
    const current = new URLSearchParams(Array.from(searchParams.entries()));
    Object.entries(newParams).forEach(([key, value]) => {
      if (value) {
        current.set(key, value);
      } else {
        current.delete(key);
      }
    });
    const query = current.toString();
    router.push(`${pathname}?${query}`);
  };


  // Fetch races based on date range and venue (adapt useRaces or API accordingly)
  // For simplicity, we'll assume useRaces can take a date range or a single date (using 'to' date for now)
  const { data: races, isLoading: isLoadingRaces, error: racesError } = useRaces(
    dateRange.to ? formatISO(dateRange.to, { representation: 'date' }) : undefined
    // In a real app, useRaces might need to accept dateRange.from, dateRange.to, and selectedVenue
  );

  const { data: predictionData, isLoading: isLoadingPrediction, error: predictionError } = useRacePrediction(selectedRaceId);

  const filteredRaces = useMemo(() => {
    if (!races) return [];
    return races.filter(race =>
      (!selectedVenue || race.venue?.toUpperCase().includes(selectedVenue.toUpperCase())) &&
      // Add date filtering if useRaces doesn't handle the full range
      (new Date(race.startTime) >= (dateRange.from || new Date(0))) &&
      (new Date(race.startTime) <= (dateRange.to || new Date()))
    );
  }, [races, selectedVenue, dateRange]);

  const raceOptions = useMemo(() => {
    return filteredRaces.map(race => ({
      value: race.id,
      label: `${race.venue || 'N/A'} - R${race.raceNumber}: ${race.name} (${new Date(race.startTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})})`,
    }));
  }, [filteredRaces]);

  return (
    <div className="space-y-6 lg:space-y-8">
      <Flex className="flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
            <Title>Horse Racing Analytics</Title>
            <Text>Select filters to find races and view detailed predictions.</Text>
        </div>
        {/* Add any global action buttons here */}
      </Flex>

      <Grid numItemsSm={1} numItemsMd={2} numItemsLg={3} className="gap-6">
        <Card>
          <Flex alignItems="center" className="mb-2">
            <CalendarIcon className="h-5 w-5 text-tremor-content-subtle mr-2" />
            <Text>Date Range</Text>
          </Flex>
          <DateRangePicker
            value={dateRange}
            onValueChange={(value) => {
              setDateRange(value);
              updateQueryParams({ from: value.from?.toISOString().split('T')[0], to: value.to?.toISOString().split('T')[0] });
              setSelectedRaceId(undefined); // Reset race
            }}
            enableSelect={false} // Allows selecting a range
            className="w-full"
          />
        </Card>
        <Card>
          <Flex alignItems="center" className="mb-2">
            <FilterIcon className="h-5 w-5 text-tremor-content-subtle mr-2" />
            <Text>Filter by Venue</Text>
          </Flex>
          <Select
            value={selectedVenue}
            onValueChange={(value) => {
              setSelectedVenue(value);
              updateQueryParams({ venue: value });
              setSelectedRaceId(undefined); // Reset race
            }}
            placeholder="All Venues"
          >
            <SelectItem value="">All Venues</SelectItem>
            {venues.map(venue => (
              <SelectItem key={venue.value} value={venue.value}>
                {venue.label}
              </SelectItem>
            ))}
          </Select>
        </Card>
        <Card>
          <Flex alignItems="center" className="mb-2">
             <ListChecksIcon className="h-5 w-5 text-tremor-content-subtle mr-2" /> {/* Placeholder Icon */}
            <Text>Select Race</Text>
          </Flex>
          {isLoadingRaces ? <Text>Loading races...</Text> : racesError ? <Text>Error: {/* @ts-ignore */}{racesError.message}</Text> : (
            <Select
              value={selectedRaceId}
              onValueChange={(value) => {
                setSelectedRaceId(value);
                updateQueryParams({ raceId: value });
              }}
              placeholder="Select a race..."
              disabled={filteredRaces.length === 0}
            >
              {raceOptions.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </Select>
          )}
          {filteredRaces.length === 0 && !isLoadingRaces && <Text className="mt-1 text-sm">No races found for selected filters.</Text>}
        </Card>
      </Grid>

      {selectedRaceId && (
        <div className="mt-6 lg:mt-8">
          {isLoadingPrediction && <Text className="text-center">Loading prediction data...</Text>}
          {predictionError && <Text className="text-center text-rose-500">Error loading prediction: {/* @ts-ignore */}{predictionError.message}</Text>}
          {predictionData && (
            <Grid numItemsSm={1} numItemsLg={2} className="gap-6">
              <Col numColSpanLg={1}>
                <HorseRacingPrediction predictions={predictionData.predictions} raceName={races?.find(r=>r.id === selectedRaceId)?.name} />
              </Col>
              <Col numColSpanLg={1}>
                <RaceAnalytics analytics={predictionData.analytics} raceName={races?.find(r=>r.id === selectedRaceId)?.name} />
              </Col>
            </Grid>
          )}
        </div>
      )}
    </div>
  );
}

// Wrap with Suspense for Next.js 14 searchParams handling
export default function HorseRacingPageContainer() {
  return (
    <Suspense fallback={<Text>Loading page filters...</Text>}>
      <HorseRacingContent />
    </Suspense>
  )
}
