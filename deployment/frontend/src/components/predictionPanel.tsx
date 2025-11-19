import React, { useState, useEffect } from 'react';
import { Drawer, DrawerContent } from "@/components/ui/drawer";
import { X } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

// Define local interface
interface HistoryDataPoint {
  year: number;
  price: number;
}

interface PredictionPanelProps {
  prediction: number | null | undefined;
  history: HistoryDataPoint[];
}

export default function PredictionPanel({ prediction, history }: PredictionPanelProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (prediction !== undefined) setOpen(true);
  }, [prediction]);

  const formatPriceDetailed = (value: number) => {
      return new Intl.NumberFormat('en-GB', {
        style: 'currency',
        currency: 'GBP',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value);
    };

  // Helper to get dynamic title
  const getChartTitle = () => {
    if (!history || history.length === 0) return "PRICE TREND";
    const start = history[0].year;
    const end = history[history.length - 1].year;
    return `PRICE TREND (${start} - ${end})`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border rounded shadow-sm text-xs">
          <p className="font-bold">{label}</p>
          <p className="text-[#ff385c]">{formatPriceDetailed(payload[0].value)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Drawer open={open} onOpenChange={setOpen}>
      <DrawerContent className="p-4 bg-white text-black shadow-xl h-[400px]"> 
        <button onClick={() => setOpen(false)} className="absolute right-4 top-2 p-2 hover:opacity-70 z-50">
            <X className="h-5 w-5" />
        </button>

        <div className="grid grid-cols-3 gap-4 py-6 h-full">
          
          {/* 1. Prediction Card */}
          <div className="p-6 rounded-xl border flex flex-col items-center justify-center bg-slate-50">
            {prediction === null && <p className="animate-pulse">Calculating...</p>}
            {prediction !== null && prediction !== undefined && (
              <>
                <p className="text-sm text-gray-500 uppercase font-bold mb-2">Estimated Value</p>
                <p className="text-3xl font-bold text-[#ff385c]">{formatPriceDetailed(prediction)}</p>
              </>
            )}
          </div>

          {/* 2. History Chart */}
          <div className="p-4 rounded-xl border col-span-2 relative">
             {/* ⭐ DYNAMIC TITLE HERE */}
             <p className="text-xs text-gray-500 font-bold absolute top-2 left-4 z-10">
                {getChartTitle()}
             </p>
             
             {history && history.length > 0 ? (
                 <div className="w-full h-full pt-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
                            <XAxis 
                                dataKey="year" 
                                tick={{fontSize: 10, fill: '#888'}} 
                                axisLine={false} 
                                tickLine={false} 
                            />
                            <YAxis 
                                tickFormatter={(value) => `£${(value/1000).toFixed(0)}k`} 
                                tick={{fontSize: 10, fill: '#888'}} 
                                axisLine={false} 
                                tickLine={false} 
                                width={40}
                            />
                            <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#ff385c', strokeWidth: 1, strokeDasharray: '4 4' }} />
                            <Line type="monotone" dataKey="price" stroke="#ff385c" strokeWidth={3} dot={false} activeDot={{ r: 6 }}/>
                        </LineChart>
                    </ResponsiveContainer>
                 </div>
             ) : (
                 <div className="flex items-center justify-center h-full text-gray-300 text-sm">
                    {prediction === null ? "Loading History..." : "No history data available"}
                 </div>
             )}
          </div>

        </div>
      </DrawerContent>
    </Drawer>
  );
}   