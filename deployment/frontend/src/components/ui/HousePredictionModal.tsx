import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

// --- Utility Function ---

/**
 * Extracts the year (YYYY) and month (MM) as numbers from a date string in YYYY-MM-DD format.
 * @param dateString The date string in "YYYY-MM-DD" format.
 * @returns An object containing the year and month as numbers (0 for invalid/missing).
 */
const extractDateComponents = (dateString: string): { sale_year: number; sale_month: number } => {
  if (dateString && dateString.includes("-")) {
    const parts = dateString.split("-");
    const yearNumber = parseInt(parts[0], 10);
    const monthNumber = parseInt(parts[1], 10);

    return {
      sale_year: isNaN(yearNumber) ? 0 : yearNumber,
      sale_month: isNaN(monthNumber) ? 0 : monthNumber,
    };
  }

  return { sale_year: 0, sale_month: 0 };
};

// --- Form Data Interface for Type Safety ---

interface FormData {
  sale_date: string;
  property_type: string;
  old_new: string;
  duration: string;
  town_city: string;
  district: string;
  county: string;
  sale_year: number;
  sale_month: number;
}


export default function HousePredictionModal() {
  const [formData, setFormData] = useState<FormData>({ // Use the interface here
    sale_date: "",
    property_type: "",
    old_new: "",
    duration: "",
    town_city: "",
    district: "",
    county: "",
    sale_year: 0,
    sale_month: 0
  })

  const [prediction, setPrediction] = useState<number>()

  /**
   * Handles form input changes. If the changed field is 'sale_date', 
   * it automatically calculates and updates 'sale_year'.
   */
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    // Create the base update object with the current field's new value
    const baseUpdate = { [name]: value };

    let dateUpdates = {};
    
    // 1. --- SPECIAL LOGIC FOR sale_date ---
    if (name === "sale_date") {
      const { sale_year, sale_month } = extractDateComponents(value);
      dateUpdates = { sale_year, sale_month };
    }

    // Merge all updates (base update + year/month updates if applicable)
    const newFormData = {
      ...formData,
      ...baseUpdate,
      ...dateUpdates
    };

    setFormData(newFormData);
    // Log the correct, updated object
    console.log("Updated Form Data:", newFormData);
  }

  const handlePredict = async () => {
    try {
        console.log("Form Data Submitted:", formData);
        
        console.log(formData)

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // The formData now includes sale_year as numbers
        body: JSON.stringify(formData),
      })
      
      if (!response.ok) {
        // Attempt to parse error response for better logging
        const errorData = await response.json();
        console.error("API Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.error || response.statusText}`);
      }

      const data = await response.json()
      
      // Check if the prediction data is a valid number before setting state
      if (typeof data.prediction === 'number' && !isNaN(data.prediction)) {
        setPrediction(data.prediction)
      } else {
        console.error("Prediction result was invalid or NaN:", data);
        console.log(data)
        setPrediction(0);
      }

    } catch (error) {
      console.error("Prediction error:", error)
    }
  }

  return (
    <Dialog>
      {/* <DialogTrigger asChild>
        <Button className="mb-4">Predict House Price</Button>
      </DialogTrigger> */}

      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Predict Housing Price</DialogTitle>
          <DialogDescription>
            Fill in the fields below to generate a price prediction.
          </DialogDescription>
        </DialogHeader>

        {/* FORM */}
        <div className="grid grid-cols-2 gap-4 mt-4">

          <input
            name="sale_date"
            type="date"
            onChange={handleChange}
            className="border p-2 rounded"
            placeholder="Sale Date"
            value={formData.sale_date} 
          />

          <select
            name="property_type"
            onChange={handleChange}
            className="border p-2 rounded"
            value={formData.property_type} 
          >
            <option value="">Property Type</option>
            <option value="D">Detached</option>
            <option value="S">Semi-Detached</option>
            <option value="T">Terraced</option>
            <option value="F">Flats</option>
            <option value="O">Other</option>
          </select>

          <select
            name="old_new"
            onChange={handleChange}
            className="border p-2 rounded"
            value={formData.old_new}
          >
            <option value="">Old/New</option>
            <option value="Y">New</option>
            <option value="N">Old</option>
          </select>

          <select
            name="duration"
            onChange={handleChange}
            className="border p-2 rounded"
            value={formData.duration}
          >
            <option value="">Duration</option>
            <option value="F">Freehold</option>
            <option value="L">Leasehold</option>
          </select>

          <input
            name="town_city"
            onChange={handleChange}
            placeholder="Town / City"
            className="border p-2 rounded"
            value={formData.town_city}
          />

          <input
            name="district"
            onChange={handleChange}
            placeholder="District"
            className="border p-2 rounded"
            value={formData.district}
          />

          <input
            name="county"
            onChange={handleChange}
            placeholder="County"
            className="border p-2 rounded"
            value={formData.county}
          />
        </div>

        {/* Predict button */}
        <Button onClick={handlePredict} className="mt-4">
          Predict
        </Button>

        {/* Show prediction */}
        {prediction !== undefined && (
          <div className="mt-4 p-3 text-lg font-bold bg-green-200 rounded">
            Predicted Price: £{Math.round(prediction).toLocaleString()}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}