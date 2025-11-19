 import React, { useState, useEffect } from 'react';

import { Drawer, DrawerContent, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";

import { X } from "lucide-react";


interface PredictionPanelProps {

  prediction: number | null | undefined;

}


export default function PredictionPanel({ prediction }: PredictionPanelProps) {

  const [open, setOpen] = useState(false);


  useEffect(() => {

    if (prediction !== undefined) setOpen(true);

  }, [prediction]);


  const formatPrice = (value: number) => {

    return new Intl.NumberFormat('en-GB', {

      style: 'currency',

      currency: 'GBP',

      minimumFractionDigits: 2,

      maximumFractionDigits: 2,

    }).format(value);

  };


  return (

    <Drawer open={open} onOpenChange={setOpen}>

      <DrawerContent className="p-6 bg-white text-black shadow-xl">

        <button onClick={() => setOpen(false)} className="absolute right-4 top-4 p-2 hover:opacity-70">

            <X className="h-5 w-5" />

          </button>


        <div className="grid grid-cols-3 gap-4 py-6 text-center">

          <div className="p-4 rounded-xl border">

            {prediction === null && <p>Loading...</p>}

            {prediction !== null && prediction !== undefined && (

              <p className="text-2xl font-bold">{formatPrice(prediction)}</p>

            )}

          </div>


          <div className="p-4 rounded-xl border">

            <p className="opacity-50">Placeholder A</p>

          </div>


          <div className="p-4 rounded-xl border">

            <p className="opacity-50">Placeholder B</p>

          </div>

        </div>

      </DrawerContent>

    </Drawer>

  );

} 