import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Interface for the new prop
interface AddressData {
    displayName: string;
    district: string;
    county: string;
    lat: number;
    lng: number;
}

interface MapProps {
    centerCity: string;
    // ⭐ NEW PROP: A callback function to pass the geocoded address data back to the parent.
    onGeocodeComplete: (data: AddressData | null) => void;
}

// Helper function to geocode a city name and update the map
const geocodeAndCenterMap = async (map: L.Map, cityName: string): Promise<AddressData | null> => {
    if (!cityName) return null;

    try {
        const countryCode = "gb"; 
        // Force English and request address details
        const api = `https://nominatim.openstreetmap.org/search?format=geojson&limit=1&city=${encodeURIComponent(cityName)}&countrycodes=${countryCode}&addressdetails=1&accept-language=en`; 

        const response = await fetch(api);
        const data = await response.json();

        if (data.features && data.features.length > 0) {
            const feature = data.features[0];
            const [lng, lat] = feature.geometry.coordinates;
            
            const displayName = feature.properties.display_name;

            const address = feature.properties.address || {};
            
            // ⭐ COUNTY LOGIC (Kept from previous fix to avoid "England")
            let extractedCounty = address.county || address.state || "";
            if (extractedCounty.toLowerCase() === 'england' || extractedCounty.toLowerCase() === 'scotland' || extractedCounty.toLowerCase() === 'wales') {
                extractedCounty = address.state_district || "";
            }
            // If the primary result was filtered (i.e., it was "England"), try state_district as a fallback for the County.
            if (extractedCounty === "") {
                extractedCounty = address.state_district || address.city_district || "";
            }
            
            // ⭐ DISTRICT FIX: Explicitly set the district to an empty string.
            const districtFinal = "";
            const countyFinal = extractedCounty.trim();

            // Clear existing markers/layers and add new marker (as before)
            map.eachLayer((layer) => {
                if (layer instanceof L.Marker) {
                    map.removeLayer(layer);
                }
            });

            const marker = L.marker([lat, lng], { title: displayName });
            marker.addTo(map).bindPopup(displayName).openPopup();
            map.setView([lat, lng], 10); 
            
            // Return the necessary address information
            return {
                displayName, 
                // ⭐ Returned District is now an empty string
                district: districtFinal, 
                county: countyFinal, 
                lat: lat, 
                lng: lng
            };
        } else {
            console.log(`No results found for city: ${cityName}`);
            return null;
        }
    } catch (err) {
        console.error("Geocoding error:", err);
        return null;
    }
}

export default function Map({ centerCity, onGeocodeComplete }: MapProps) {
    const mapRef = useRef<L.Map | null>(null);

    // 1. Map Initialization (Runs once)
    useEffect(() => {
        const config = { 
            minZoom: 2, 
            maxZoom: 18,
            zoomControl: false, 
            dragging: false,
        };
        const zoom = 5;
        const lat = 54.0; // Initial UK center
        const lng = -2.0;

        const mapElement = document.getElementById("map");
        if (!mapElement || mapRef.current) return;

        const map = L.map("map", config).setView([lat, lng], zoom);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        }).addTo(map);

        L.control.zoom({ position: 'bottomleft' }).addTo(map);

        mapRef.current = map;

        // Cleanup function for map destruction
        return () => {
            if (mapRef.current) {
                mapRef.current.remove();
                mapRef.current = null;
            }
        };
    }, []);

    // 2. City Search/Centering (Runs whenever centerCity changes)
    useEffect(() => {
        if (mapRef.current && centerCity) {
            // Define the async function inside the effect
            const runGeocoding = async () => {
                // Call the helper and wait for the address data
                const locationData = await geocodeAndCenterMap(mapRef.current!, centerCity);
                
                // ⭐ NEW: Send the data back to the parent component
                onGeocodeComplete(locationData);
            }

            // Simple debounce logic
            const handler = setTimeout(() => {
                runGeocoding();
            }, 500); 

            return () => {
                clearTimeout(handler);
            };
        }
        // Run whenever centerCity or the callback changes
    }, [centerCity, onGeocodeComplete]);

    return (
        <div id="map" className="absolute inset-0 w-full h-full z-0" />
    );
}