"use client";

import React, { useState, useRef, useEffect } from "react";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ⭐ 1. Define locally (Do NOT export to avoid circular dependency bugs)
interface HistoryDataPoint {
  year: number;
  price: number;
}

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

const getDisplayValue = (field: keyof FormData, value: string): string => {
  if (!value) return "";
  switch (field) {
    case "property_type":
      return { D: "Detached", S: "Semi", T: "Terraced", F: "Flat", O: "Other" }[value] || value;
    case "old_new":
      return { Y: "New", N: "Old" }[value] || value;
    case "duration":
      return { F: "Freehold", L: "Leasehold" }[value] || value;
    default:
      return value;
  }
};

interface FormData {
  date_of_transfer: string;
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

type ActiveSection = keyof Omit<FormData, "sale_year" | "sale_month"> | null;
type AddressFillerFunction = (data: AddressData) => void;

interface AddressData {
  district: string;
  county: string;
}

interface HousePredictionSearchBarProps {
  onCityChange: (city: string) => void;
  onAddressFill: (fillerFunc: AddressFillerFunction) => void;
  onPredictionMade: (price: number | null, history?: HistoryDataPoint[]) => void;
}

export default function HousePredictionSearchBar({ onCityChange, onAddressFill, onPredictionMade }: HousePredictionSearchBarProps) {
  const [formData, setFormData] = useState<FormData>({
    date_of_transfer: "",
    sale_date: "",
    property_type: "",
    old_new: "",
    duration: "",
    town_city: "",
    district: "",
    county: "",
    sale_year: 0,
    sale_month: 0,
  });

  const [activeSection, setActiveSection] = useState<ActiveSection>(null);

  const fillAddressFields = (data: AddressData) => {
    setFormData(prevData => ({
      ...prevData,
      district: data.district,
      county: data.county,
    }));
  };

  useEffect(() => {
    onAddressFill(fillAddressFields);
  }, [onAddressFill]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const baseUpdate = { [name]: value };
    let dateUpdates = {};
    if (name === "sale_date") {
      const { sale_year, sale_month } = extractDateComponents(value);
      dateUpdates = { 
        sale_year,
        sale_month,
        date_of_transfer: value
      };
    }
    const newFormData = { ...formData, ...baseUpdate, ...dateUpdates };
    setFormData(newFormData);

    if (name === "town_city") {
      onCityChange(value);
    }
  };

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL; // ✅ usa env

  const handlePredict = async () => {
    onPredictionMade(null, []); // Reset

    try {
      console.log("Form Data Submitted:", formData);

      const [predResponse, histResponse] = await Promise.all([
        fetch(`${BACKEND_URL}/housing/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        }),
        fetch(`${BACKEND_URL}/housing/history`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        })
      ]);

      if (!predResponse.ok) throw new Error("Prediction failed");
      const predData = await predResponse.json();
      const finalPrice = (typeof predData.prediction === 'number' && !isNaN(predData.prediction)) 
          ? predData.prediction 
          : 0;

      let finalHistory: HistoryDataPoint[] = [];
      if (histResponse.ok) {
        const histData = await histResponse.json();
        if (histData.history && Array.isArray(histData.history)) {
          finalHistory = histData.history;
        }
      }

      onPredictionMade(finalPrice, finalHistory);

    } catch (error) {
      console.error("Prediction error:", error);
      onPredictionMade(0, []);
    }
  };

  const FieldSegment: React.FC<{
    label: string;
    fieldKey: ActiveSection;
    inputType: "text" | "date" | "select";
  }> = ({ label, fieldKey, inputType }) => {
    const inputRef = useRef<HTMLInputElement | HTMLSelectElement>(null);
    const isActive = activeSection === fieldKey;
    const value = formData[fieldKey] as string;
    const hasValue = !!value;

    useEffect(() => {
      if (isActive && inputRef.current) {
        setTimeout(() => inputRef.current?.focus(), 50);
        if (inputType === "select") {
          const selectEl = inputRef.current as HTMLSelectElement;
          setTimeout(() => selectEl.showPicker?.(), 60);
        }
        if (inputType === "date") {
          const dateEl = inputRef.current as HTMLInputElement;
          setTimeout(() => dateEl.showPicker?.(), 60);
        }
      }
    }, [isActive]);

    const handleSegmentClick = () => setActiveSection(fieldKey);

    const renderInput = () => {
      const commonProps = {
        name: fieldKey,
        onChange: handleChange,
        ref: inputRef,
        className: "w-full bg-transparent text-sm font-bold outline-none z-10 p-0",
        value: value,
      };

      switch (inputType) {
        case "date":
          return <input type="date" {...commonProps} />;
        case "select":
          return (
            <select {...commonProps} className={cn(commonProps.className, "appearance-none cursor-pointer h-full")}>
              <option value="" disabled hidden>{label}</option>
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
          );
        default:
          const isCountyReadOnly = fieldKey === "county" && hasValue;
          return <input type="text" aria-label={label} {...commonProps} readOnly={isCountyReadOnly} />;
      }
    };

    return (
      <div
        className={cn(
          "relative flex-shrink-0 cursor-pointer rounded-full p-4 h-[68px] flex flex-col justify-center transition-all duration-200 ease-in-out z-40",
          isActive ? "bg-background scale-[1.03]" : "hover:bg-gray-100 scale-100"
        )}
        onClick={handleSegmentClick}
      >
        {isActive || (inputType !== "text" && hasValue) ? (
          <span className="text-xs font-bold uppercase text-gray-600 mb-1">{label}</span>
        ) : (
          <span className="text-sm text-gray-700 font-semibold mb-1">{label}</span>
        )}

        <div className="flex items-center pt-1">
          {isActive ? renderInput() : hasValue ? (
            <span className="text-sm font-bold text-foreground">{getDisplayValue(fieldKey, value)}</span>
          ) : (
            <span className="text-sm text-muted-foreground">{label}</span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full px-4 pt-1.5 relative z-[1000]">
      <div
        className="relative flex items-center justify-between rounded-full border bg-background w-full max-w-7xl mx-auto p-1 shadow-md"
        onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setActiveSection(null); }}
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
    </div>
  );
}