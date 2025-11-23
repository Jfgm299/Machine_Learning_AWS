"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem
} from "@/components/ui/select";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from "recharts";

interface ElectricityInput {
  SETTLEMENT_DATE: string | null;
  SETTLEMENT_PERIOD: number | null;
  TSD: number;

  EMBEDDED_WIND_GENERATION: number;
  EMBEDDED_WIND_CAPACITY: number;
  EMBEDDED_SOLAR_GENERATION: number;
  EMBEDDED_SOLAR_CAPACITY: number;

  NON_BM_STOR: number;
  PUMP_STORAGE_PUMPING: number;
  SCOTTISH_TRANSFER: number;
  IFA_FLOW: number;
}

export default function ElectricityPage() {
  const [started, setStarted] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [monthlyData, setMonthlyData] = useState<{ date: string; mean: number }[]>([]);
  const [loadingMonthly, setLoadingMonthly] = useState(false);

  const [inputs, setInputs] = useState<ElectricityInput>({
    SETTLEMENT_DATE: null,
    SETTLEMENT_PERIOD: null,
    TSD: 0,
    EMBEDDED_WIND_GENERATION: 0,
    EMBEDDED_WIND_CAPACITY: 0,
    EMBEDDED_SOLAR_GENERATION: 0,
    EMBEDDED_SOLAR_CAPACITY: 0,
    NON_BM_STOR: 0,
    PUMP_STORAGE_PUMPING: 0,
    SCOTTISH_TRANSFER: 0,
    IFA_FLOW: 0,
  });

  const [toggles, setToggles] = useState({
    solarGen: false,
    windGen: false,
    solarCap: false,
    windCap: false,
  });

  // Log every update
  useEffect(() => {
    console.log("🔄 Updated Electricity Inputs:", JSON.stringify(inputs, null, 2));
  }, [inputs]);

  const updateField = (field: keyof ElectricityInput, value: any) => {
    const newObject = { ...inputs, [field]: value };

    console.log("🔄 Input Changed → new backend object:");
    console.log(JSON.stringify(newObject, null, 2));

    setInputs(newObject);
  };

  const handleToggle = (
    toggleKey: keyof typeof toggles,
    field: keyof ElectricityInput
  ) => {
    const newToggles = { ...toggles, [toggleKey]: !toggles[toggleKey] };
    setToggles(newToggles);

    if (toggles[toggleKey] === true) {
      const newObject = { ...inputs, [field]: 0 };

      console.log("🔄 Toggle OFF → backend object:");
      console.log(JSON.stringify(newObject, null, 2));

      setInputs(newObject);
    }
  };

  // Fetch single prediction + monthly means
  const handleCalculate = async () => {
    const finalObject = { ...inputs };
    console.log("FINAL OBJECT →", finalObject);

    if (!finalObject.SETTLEMENT_DATE || finalObject.SETTLEMENT_PERIOD === null) {
      console.error("❌ Missing required fields: date or period");
      return;
    }

    try {
      // 1) Single prediction (same as before)
      const resp1 = await fetch("http://localhost:8000/electricity/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalObject),
      });
      const data1 = await resp1.json();
      console.log("⚡ PREDICTED ND →", data1.prediction);
      setPrediction(data1.prediction);

      // 2) Monthly averages (daily mean across the month)
      setLoadingMonthly(true);
      const resp2 = await fetch("http://localhost:8000/electricity/monthly", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalObject),
      });
      const data2 = await resp2.json();
      if (data2.monthly) {
        // convert to recharts-friendly array: {date, mean}
        setMonthlyData(data2.monthly.map((d: any) => ({ date: d.date, mean: Number(d.mean) })));
      } else {
        setMonthlyData([]);
      }
      setLoadingMonthly(false);

    } catch (error) {
      console.error("🔥 Error calling backend:", error);
      setLoadingMonthly(false);
    }
  };

  // Small helper to format date labels (MM-DD)
  const shortDate = (iso: string) => {
    try {
      const dt = new Date(iso + "T00:00:00");
      return `${dt.getDate()}/${dt.getMonth() + 1}`;
    } catch {
      return iso;
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-screen bg-slate-50 dark:bg-slate-950">

      {!started && (
        <div className="flex flex-col items-center gap-4">
          <h1 className="text-4xl font-bold text-slate-800 dark:text-white">
            National Demand Calculator
          </h1>

          <Button className="px-6 py-3 text-lg" onClick={() => setStarted(true)}>
            Start
          </Button>
        </div>
      )}

      {started && (
        <>
        <Card className="w-full max-w-5xl mt-6 border border-slate-300 dark:border-slate-700 shadow-xl mb-6">
          <CardHeader>
            <CardTitle className="text-2xl font-semibold">
              Input Parameters
            </CardTitle>
          </CardHeader>

          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              <div className="space-y-6">
                
                {/* SETTLEMENT DATE, SETTLEMENT PERIOD, TSD */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  
                  <div className="flex flex-col space-y-2">
                    <Label>Settlement Date</Label>
                    <Input
                      type="date"
                      className="bg-slate-100 dark:bg-slate-800 h-10"
                      onChange={(e) => updateField("SETTLEMENT_DATE", e.target.value)}
                    />
                  </div>
                  
                  <div className="flex flex-col space-y-2">
                    <Label>Settlement Period</Label>
                    <Select
                      onValueChange={(value) => updateField("SETTLEMENT_PERIOD", Number(value))}
                    >
                      <SelectTrigger className="bg-slate-100 dark:bg-slate-800 h-10">
                        <SelectValue placeholder="Select period" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 48 }, (_, i) => i + 1).map((num) => (
                          <SelectItem key={num} value={String(num)}>
                            {num}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex flex-col space-y-2">
                    <Label>TSD</Label>
                    <Input
                      type="number"
                      className="bg-slate-100 dark:bg-slate-800 h-10"
                      onChange={(e) => updateField("TSD", Number(e.target.value))}
                    />
                  </div>
                </div>

                {/* Generations + Capacity */}
                <div className="border rounded-lg p-4 bg-slate-100 dark:bg-slate-900 border-slate-300 dark:border-slate-700">
                  <h3 className="font-bold mb-3">Generations</h3>

                  <div className="flex items-center gap-3 mb-2">
                    <Checkbox
                      checked={toggles.solarGen}
                      onCheckedChange={() => handleToggle("solarGen", "EMBEDDED_SOLAR_GENERATION")}
                    />
                    <Label>Embedded Solar Generation</Label>
                  </div>
                  {toggles.solarGen && (
                    <Input
                      type="number"
                      className="bg-white dark:bg-slate-800 mb-3"
                      onChange={(e) => updateField("EMBEDDED_SOLAR_GENERATION", Number(e.target.value))}
                    />
                  )}

                  <div className="flex items-center gap-3 mb-2">
                    <Checkbox
                      checked={toggles.windGen}
                      onCheckedChange={() => handleToggle("windGen", "EMBEDDED_WIND_GENERATION")}
                    />
                    <Label>Embedded Wind Generation</Label>
                  </div>
                  {toggles.windGen && (
                    <Input
                      type="number"
                      className="bg-white dark:bg-slate-800 mb-3"
                      onChange={(e) => updateField("EMBEDDED_WIND_GENERATION", Number(e.target.value))}
                    />
                  )}

                  <h4 className="font-semibold mt-3 mb-2">Capacity</h4>

                  <div className="flex items-center gap-3 mb-2">
                    <Checkbox
                      checked={toggles.solarCap}
                      onCheckedChange={() => handleToggle("solarCap", "EMBEDDED_SOLAR_CAPACITY")}
                    />
                    <Label>Embedded Solar Capacity</Label>
                  </div>
                  {toggles.solarCap && (
                    <Input
                      type="number"
                      className="bg-white dark:bg-slate-800 mb-3"
                      onChange={(e) => updateField("EMBEDDED_SOLAR_CAPACITY", Number(e.target.value))}
                    />
                  )}

                  <div className="flex items-center gap-3 mb-2">
                    <Checkbox
                      checked={toggles.windCap}
                      onCheckedChange={() => handleToggle("windCap", "EMBEDDED_WIND_CAPACITY")}
                    />
                    <Label>Embedded Wind Capacity</Label>
                  </div>
                  {toggles.windCap && (
                    <Input
                      type="number"
                      className="bg-white dark:bg-slate-800"
                      onChange={(e) => updateField("EMBEDDED_WIND_CAPACITY", Number(e.target.value))}
                    />
                  )}
                </div>
              </div>

              {/* Others */}
              <div className="border rounded-lg p-4 bg-slate-100 dark:bg-slate-900 border-slate-300 dark:border-slate-700">
                <h3 className="font-bold mb-4">Others</h3>
                {[
                  "NON_BM_STOR",
                  "PUMP_STORAGE_PUMPING",
                  "SCOTTISH_TRANSFER",
                  "IFA_FLOW"
                ].map((field) => (
                  <div key={field} className="flex flex-col mb-4">
                    <Label>{field.replace(/_/g, " ")}</Label>
                    <Input
                      type="number"
                      className="bg-white dark:bg-slate-800"
                      onChange={(e) =>
                        updateField(field as keyof ElectricityInput, Number(e.target.value))
                      }
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-8 flex justify-end">
              <Button className="px-6 py-3 text-lg" onClick={handleCalculate}>
                Calculate
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Prediction Card (shows after prediction) */}
        {prediction !== null && (
          <Card className="w-full max-w-5xl mb-10 border border-slate-300 dark:border-slate-700 shadow-lg">
            <CardContent>
              <div className="flex flex-col md:flex-row gap-4 items-stretch">
                {/* Left: big number */}
                <div className="flex-1 flex items-center justify-center p-6 bg-gradient-to-br from-white to-slate-50 dark:from-slate-900 dark:to-slate-800 rounded-lg">
                  <div className="text-center">
                    <div className="text-sm font-medium text-muted-foreground">Predicted ND</div>
                    <div className="mt-2 text-5xl md:text-6xl font-extrabold tracking-tight text-rose-600 dark:text-rose-400">
                      {prediction.toFixed(0)}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">MW</div>
                  </div>
                </div>

                {/* Right: line chart for monthly means */}
                <div className="flex-1 p-4 rounded-lg bg-white dark:bg-slate-900">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold">Monthly trend (daily mean)</h4>
                    {loadingMonthly && <div className="text-sm text-muted-foreground">loading…</div>}
                  </div>

                  {monthlyData.length === 0 ? (
                    <div className="text-sm text-muted-foreground">No monthly data</div>
                  ) : (
                    <div style={{ width: "100%", height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={monthlyData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tickFormatter={shortDate} />
                          <YAxis domain={[25000, "dataMax" + 2000]} />  {/* <-- Cambio aquí */}
                          <Tooltip labelFormatter={(label) => `Date: ${label}`} />
                          <Line type="monotone" dataKey="mean" stroke="#ff385c" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        </>
      )}
    </div>
  );
}