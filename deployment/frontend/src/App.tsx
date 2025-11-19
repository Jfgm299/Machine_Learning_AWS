import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import NavBar from "./components/navbar"
import HousePredictionModal from "@/components/ui/HousePredictionModal"

// Páginas nuevas
import HousingPage from "./pages/HousingPage"
import ElectricityPage from "./pages/ElectricityPage"

function App() {
  return (
    <Router>
      <div className="justify-center p-6">
        <NavBar />

        <Routes>
          <Route path="/" element={<HousePredictionModal />} />
          <Route path="/housing" element={<HousingPage />} />
          <Route path="/electricity" element={<ElectricityPage />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App