import { useState } from "react";
import HousePredictionSearchBar from "@/components/searchbar"
import PredictionPanel from "@/components/predictionPanel";
import Map from "@/components/map"
// Note: Card components are not used here, but kept in imports for completeness
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"

// Define interfaces for type safety, matching the Map and SearchBar props
interface AddressData {
    district: string;
    county: string;
}

// Define the type for the function that sets District/County state in the SearchBar
type AddressFillerFunction = (data: AddressData) => void;

// Define the type for the data returned by the geocoding in the Map
interface LocationData {
    displayName: string;
    district: string;
    county: string;
    lat: number;
    lng: number;
}


export default function HousingPage() {
    const [searchCity, setSearchCity] = useState("");
    const [predictionPrice, setPredictionPrice] = useState<number | null>(undefined); // Use undefined to hide initially
    // ⭐ 1. STATE TO HOLD THE SEARCH BAR'S ADDRESS FILLER FUNCTION
    const [addressFiller, setAddressFiller] = useState<AddressFillerFunction | null>(null);

    // Function passed to the SearchBar to capture its internal state setter function
    const handleSetAddressFiller = (fillerFunc: AddressFillerFunction) => {
        setAddressFiller(() => fillerFunc);
    };

    const handlePredictionMade = (price: number | null) => {
        // null means loading/clearing; 0 means error; > 0 means success
        setPredictionPrice(price);
    };

    // Function passed to the Map. It receives the geocoded data.
    const handleGeocodeComplete = (locationData: LocationData | null) => {
        if (locationData && addressFiller) {
            // ⭐ 2. USE THE SAVED FILLER FUNCTION TO UPDATE THE SEARCH BAR'S STATE
            addressFiller({
                district: locationData.district,
                county: locationData.county,
            });
            console.log(`Geocode success: Auto-filling District: ${locationData.district}, County: ${locationData.county}`);
        } else if (!locationData) {
            // If geocoding fails, clear the fields
            if (addressFiller) {
                addressFiller({ district: "", county: "" });
            }
            console.log("Geocoding failed or returned no results for the city.");
        }
    };

    const handleCityChange = (city: string) => {
        setSearchCity(city);
    };

    return (
        <div className="relative flex justify-center items-center h-screen w-screen bg-slate-50 dark:bg-slate-950">
            
            {/* 1. Map Component (Background) */}
            <Map 
                centerCity={searchCity} 
                // ⭐ PROP 1: The Map reports its results here
                onGeocodeComplete={handleGeocodeComplete}
            /> 

            {/* 2. The Search Bar Container (Foreground) */}
            <div 
                className="absolute top-5 z-40 w-full max-w-7xl px-5"
            >
                <HousePredictionSearchBar 
                    onCityChange={handleCityChange} 
                    onAddressFill={handleSetAddressFiller} 
                    onPredictionMade={handlePredictionMade}
                />
            </div>

            <PredictionPanel prediction={predictionPrice} />
        </div>
    )
}