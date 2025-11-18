// src/components/Map.jsx
import { useEffect } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
// If using npm install for autocomplete
// import Autocomplete from "autocomplete-js";
// OR dynamically load it via CDN

export default function Map() {
  useEffect(() => {
    // --- 1. Initialize map ---
    const config = { minZoom: 6, maxZoom: 18 };
    const zoom = 3;
    const lat = 52.22977;
    const lng = 21.01178;

    const map = L.map("map", config).setView([lat, lng], zoom);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    // --- 2. Dynamically load the Autocomplete script if needed ---
    const script = document.createElement("script");
    script.src =
      "https://cdn.jsdelivr.net/gh/tomickigrzegorz/autocomplete@2.0.2/dist/js/autocomplete.min.js";
    script.onload = () => {
      // eslint-disable-next-line no-undef
      new Autocomplete("search", {
        selectFirst: true,
        howManyCharacters: 2,
        onSearch: ({ currentValue }) => {
          const api = `https://nominatim.openstreetmap.org/search?format=geojson&limit=5&city=${encodeURI(
            currentValue
          )}`;
          return fetch(api)
            .then((res) => res.json())
            .then((data) => data.features)
            .catch((err) => console.error(err));
        },
        onResults: ({ currentValue, matches, template }) => {
          const regex = new RegExp(currentValue, "gi");
          return matches === 0
            ? template
            : matches
                .map(
                  (el) => `
            <li class="loupe">
              <p>${el.properties.display_name.replace(
                regex,
                (str) => `<b>${str}</b>`
              )}</p>
            </li>`
                )
                .join("");
        },
        onSubmit: ({ object }) => {
          map.eachLayer((layer) => {
            if (layer.toGeoJSON) map.removeLayer(layer);
          });

          const { display_name } = object.properties;
          const [lng, lat] = object.geometry.coordinates;
          const marker = L.marker([lat, lng], { title: display_name });
          marker.addTo(map).bindPopup(display_name);
          map.setView([lat, lng], 8);
        },
        onSelectedItem: ({ index, element, object }) => {
          console.log("Selected:", index, object);
        },
        noResults: ({ currentValue, template }) =>
          template(`<li>No results found: "${currentValue}"</li>`),
      });
    };
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
      map.remove();
    };
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <div className="auto-search-wrapper m-2">
        <input
          type="text"
          id="search"
          autoComplete="off"
          className="full-width p-2 border border-gray-300 rounded w-full"
          placeholder="Enter the city name"
        />
      </div>
      <div
        id="map"
        className="w-full h-[600px] max-w-[1400px] mx-auto rounded shadow"
      />
    </div>
  );
}