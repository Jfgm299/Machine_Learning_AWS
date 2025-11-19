import { useState } from "react";
import HousePredictionSearchBar from "@/components/searchbar" 
import PredictionPanel from "@/components/predictionPanel";
import Map from "@/components/map"

// ⭐ 1. Define locally again
interface HistoryDataPoint {
  year: number;
  price: number;
}

interface AddressData {
    district: string;
    county: string;
}

type AddressFillerFunction = (data: AddressData) => void;

interface LocationData {
    displayName: string;
    district: string;
    county: string;
    lat: number;
    lng: number;
}

export default function HousingPage() {
    const [searchCity, setSearchCity] = useState("");
    const [predictionPrice, setPredictionPrice] = useState<number | null>(undefined);
    
    // ⭐ 2. Use local type
    const [priceHistory, setPriceHistory] = useState<HistoryDataPoint[]>([]); 
    
    const [addressFiller, setAddressFiller] = useState<AddressFillerFunction | null>(null);

    const handleSetAddressFiller = (fillerFunc: AddressFillerFunction) => {
        setAddressFiller(() => fillerFunc);
    };

    // ⭐ 3. Typescript will accept this because the structures match
    const handlePredictionMade = (price: number | null, history: HistoryDataPoint[] = []) => {
        setPredictionPrice(price);
        setPriceHistory(history);
    };

    const handleGeocodeComplete = (locationData: LocationData | null) => {
        if (locationData && addressFiller) {
            addressFiller({
                district: locationData.district,
                county: locationData.county,
            });
        } else if (!locationData) {
            if (addressFiller) {
                addressFiller({ district: "", county: "" });
            }
        }
    };

    const handleCityChange = (city: string) => {
        setSearchCity(city);
    };

    return (
        <div className="relative flex justify-center items-center h-screen w-screen bg-slate-50 dark:bg-slate-950">
            <Map 
                centerCity={searchCity} 
                onGeocodeComplete={handleGeocodeComplete}
            /> 

            <div className="absolute top-5 z-40 w-full max-w-7xl px-5">
                <HousePredictionSearchBar 
                    onCityChange={handleCityChange} 
                    onAddressFill={handleSetAddressFiller} 
                    onPredictionMade={handlePredictionMade}
                />
            </div>

            <PredictionPanel prediction={predictionPrice} history={priceHistory} />
        </div>
    )
}