import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import NavBar from "./components/navbar"
import HousePredictionModal from "@/components/ui/HousePredictionModal"

// Páginas nuevas
import HousingPage from "./pages/HousingPage"
import ElectricityPage from "./pages/ElectricityPage"
import InfoPage from "./pages/info"

function App() {
  return (
    <Router>
      <div className="justify-center">
        <NavBar />

        <Routes>
          <Route path="/" element={<HousePredictionModal />} />
          <Route path="/housing" element={<HousingPage />} />
          <Route path="/electricity" element={<ElectricityPage />} />
          <Route path="/info" element={<InfoPage />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App