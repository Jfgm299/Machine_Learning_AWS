import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

export default function HousePredictionModal() {
  const [formData, setFormData] = useState({
    sale_date: "",
    property_type: "",
    old_new: "",
    duration: "",
    town_city: "",
    district: "",
    county: "",
    record_status___monthly_file_only: "",
    sale_year: "",
  })

  const [prediction, setPrediction] = useState<number | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handlePredict = async () => {
    try {
        console.log("Form Data Submitted:", formData);
        const formattedDate = new Date(formData.sale_date).toISOString().split("T")[0];


      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()
      setPrediction(data.prediction)
    } catch (error) {
      console.error("Prediction error:", error)
    }
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button className="mb-4">Predict House Price</Button>
      </DialogTrigger>

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
          />

          <select
            name="property_type"
            onChange={handleChange}
            className="border p-2 rounded"
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
          >
            <option value="">Old/New</option>
            <option value="Y">New</option>
            <option value="N">Old</option>
          </select>

          <select
            name="duration"
            onChange={handleChange}
            className="border p-2 rounded"
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
          />

          <input
            name="district"
            onChange={handleChange}
            placeholder="District"
            className="border p-2 rounded"
          />

          <input
            name="county"
            onChange={handleChange}
            placeholder="County"
            className="border p-2 rounded"
          />

          <input
            name="record_status___monthly_file_only"
            onChange={handleChange}
            placeholder="Record Status"
            className="border p-2 rounded"
          />

          <input
            name="sale_year"
            type="number"
            onChange={handleChange}
            placeholder="Sale Year"
            className="border p-2 rounded"
          />
        </div>

        {/* Predict button */}
        <Button onClick={handlePredict} className="mt-4">
          Predict
        </Button>

        {/* Show prediction */}
        {prediction !== null && (
          <div className="mt-4 p-3 text-lg font-bold bg-green-200 rounded">
            Predicted Price: £{Math.round(prediction).toLocaleString()}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}