// Updated component with fixes: single-click open for date/select, segment collapse on blur,
// and select/date fields show the selected value instead of floating label.

import React, { useState, useRef, useEffect } from "react"
import { Search, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const extractDateComponents = (dateString: string): { sale_year: number; sale_month: number } => {
  if (dateString && dateString.includes("-")) {
    const parts = dateString.split("-")
    const yearNumber = parseInt(parts[0], 10)
    const monthNumber = parseInt(parts[1], 10)

    return {
      sale_year: isNaN(yearNumber) ? 0 : yearNumber,
      sale_month: isNaN(monthNumber) ? 0 : monthNumber,
    }
  }

  return { sale_year: 0, sale_month: 0 }
}

const getDisplayValue = (field: keyof FormData, value: string): string => {
  if (!value) return ""
  switch (field) {
    case "property_type":
      return { D: "Detached", S: "Semi", T: "Terraced", F: "Flat", O: "Other" }[value] || value
    case "old_new":
      return { Y: "New", N: "Old" }[value] || value
    case "duration":
      return { F: "Freehold", L: "Leasehold" }[value] || value
    default:
      return value
  }
}

interface FormData {
  sale_date: string
  property_type: string
  old_new: string
  duration: string
  town_city: string
  district: string
  county: string
  sale_year: number
  sale_month: number
}

type ActiveSection = keyof Omit<FormData, "sale_year" | "sale_month"> | null
type AddressFillerFunction = (data: AddressData) => void;

interface AddressData {
    district: string;
    county: string;
}

interface HousePredictionSearchBarProps {
    onCityChange: (city: string) => void;
    // ⭐ FIX: This prop must accept the function type AddressFillerFunction
    onAddressFill: (fillerFunc: AddressFillerFunction) => void; 
}

export default function HousePredictionSearchBar({ onCityChange, onAddressFill }: HousePredictionSearchBarProps) {
  const [formData, setFormData] = useState<FormData>({
    sale_date: "",
    property_type: "",
    old_new: "",
    duration: "",
    town_city: "",
    district: "",
    county: "",
    sale_year: 0,
    sale_month: 0,
  })

  const [activeSection, setActiveSection] = useState<ActiveSection>(null)

  const [prediction, setPrediction] = useState<number>()
  
    /**
     * ⭐ NEW FUNCTION: Allows parent component to automatically fill district and county.
     */
    const fillAddressFields = (data: AddressData) => {
        setFormData(prevData => ({
            ...prevData,
            district: data.district,
            county: data.county,
        }));
    };

    // Propagate the fill function to the parent via the provided prop
    useEffect(() => {
        // This ensures the parent can call fillAddressFields when geocoding is complete.
        // We pass the function reference up only once.
        onAddressFill(fillAddressFields); 
    }, [onAddressFill]);

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

      // 2. --- NOTIFY PARENT ON CITY CHANGE ---
      if (name === "town_city") {
        onCityChange(value);
      }
    }
  
    const handlePredict = async () => {
      try {
          console.log("Form Data Submitted:", formData);
          
          console.log(formData)
  
        const response = await fetch("http://localhost:8000/housing/predict", {
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

  const FieldSegment: React.FC<{
    label: string
    fieldKey: ActiveSection
    inputType: "text" | "date" | "select"
  }> = ({ label, fieldKey, inputType }) => {
    const inputRef = useRef<HTMLInputElement | HTMLSelectElement>(null)
    const isActive = activeSection === fieldKey
    const value = formData[fieldKey] as string
    const hasValue = !!value

    useEffect(() => {
      if (isActive && inputRef.current) {
        setTimeout(() => inputRef.current?.focus(), 50)
        if (inputType === "select") {
          const selectEl = inputRef.current as HTMLSelectElement
          setTimeout(() => selectEl.showPicker?.(), 60)
        }
        if (inputType === "date") {
          const dateEl = inputRef.current as HTMLInputElement
          setTimeout(() => dateEl.showPicker?.(), 60)
        }
      }
    }, [isActive])

    const handleSegmentClick = () => {
      setActiveSection(fieldKey)
    }

    const renderInput = () => {
      const commonProps = {
        name: fieldKey,
        onChange: handleChange,
        ref: inputRef,
        // Ensure readOnly for district/county fields to prevent manual editing after city search
        readOnly: (fieldKey === "county") && hasValue, 
        className: "w-full bg-transparent text-sm font-bold outline-none z-10 p-0",
        value: value,
      }

      switch (inputType) {
        case "date":
          return <input type="date" {...commonProps} />
        case "select":
          return (
            <select
              {...commonProps}
              className={cn(commonProps.className, "appearance-none cursor-pointer h-full")}
            >
              <option value="" disabled hidden>
                {label}
              </option>
              {fieldKey === "property_type" && (
                <>
                  <option value="D">Detached</option>
                  <option value="S">Semi-Detached</option>
                  <option value="T">Terraced</option>
                  <option value="F">Flats</option>
                  <option value="O">Other</option>
                </>
              )}
              {fieldKey === "old_new" && (
                <>
                  <option value="Y">New</option>
                  <option value="N">Old</option>
                </>
              )}
              {fieldKey === "duration" && (
                <>
                  <option value="F">Freehold</option>
                  <option value="L">Leasehold</option>
                </>
              )}
            </select>
          )
        default:
          return <input type="text" aria-label={label} {...commonProps} />
      }
    }

    return (
      <div
        className={cn(
          "relative flex-shrink-0 cursor-pointer rounded-full p-4 h-[68px] flex flex-col justify-center transition-all duration-200 ease-in-out z-40",
          isActive ? "bg-background scale-[1.03]" : "hover:bg-gray-100 scale-100"
        )}
        onClick={handleSegmentClick}
      >
        {/* If it's date or select AND has value, replace the label fully with selected value */}
        {inputType !== "text" && hasValue ? (
          <span className="text-xs font-bold uppercase text-gray-600 mb-1">{label}</span>
        ) : (
          <span className="text-sm text-gray-700 font-semibold mb-1">{label}</span>
        )}

        <div className="flex items-center pt-1">
          {isActive ? (
            renderInput()
          ) : hasValue ? (
            <span className="text-sm font-bold text-foreground">{getDisplayValue(fieldKey, value)}</span>
          ) : (
            <span className="text-sm text-muted-foreground">{label}</span>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="w-full px-4 pt-1.5">
      <div
        className="relative flex items-center justify-between rounded-full border bg-background w-full max-w-7xl mx-auto p-1"
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget)) {
            setActiveSection(null)
          }
        }}
        tabIndex={0}
      >
        <div className="flex flex-1 overflow-x-auto whitespace-nowrap scrollbar-hide">
          <FieldSegment label="TOWN / CITY" fieldKey="town_city" inputType="text" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="DISTRICT" fieldKey="district" inputType="text" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="COUNTY" fieldKey="county" inputType="text" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="SALE DATE" fieldKey="sale_date" inputType="date" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="PROPERTY TYPE" fieldKey="property_type" inputType="select" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="DURATION" fieldKey="duration" inputType="select" />
          <span className="h-10 w-px bg-border my-auto mx-2" />
          <FieldSegment label="OLD / NEW" fieldKey="old_new" inputType="select" />
        </div>

        <div className="p-2 flex items-center flex-shrink-0">
          <Button
            onClick={handlePredict}
            className="flex items-center gap-2 rounded-full bg-[#ff385c] px-8 py-6 font-bold text-white active:scale-95 hover:bg-[#e00b41]"
          >
            <Search className="size-5" />
            <span>Predict</span>
          </Button>
        </div>
      </div>

      {prediction !== undefined && (
        <div className="mt-4 p-4 max-w-7xl mx-auto text-lg font-bold bg-green-100 border border-green-300 text-green-800 rounded-lg shadow-inner flex items-center justify-between">
          <span>
            <CheckCircle className="size-5 inline mr-2 align-text-bottom" /> Predicted Price:
          </span>
          <span className="text-2xl">£{Math.round(prediction).toLocaleString()}</span>
        </div>
      )}
    </div>
  )
}