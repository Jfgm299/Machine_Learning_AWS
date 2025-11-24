// 1. We added 'Navigate' to the imports
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import NavBar from "./components/navbar"
import HousePredictionModal from "@/components/ui/HousePredictionModal"

import HousingPage from "./pages/HousingPage"
import ElectricityPage from "./pages/ElectricityPage"
import InfoPage from "./pages/info"

function App() {
  return (
    <Router>
      <div className="justify-center">
        <NavBar />
        <Routes>
          {/* 2. AUTOMATIC REDIRECT: If user hits "/", send them to "/housing" */}
          <Route path="/" element={<Navigate to="/housing" replace />} />
          
          <Route path="/housing" element={<HousingPage />} />
          <Route path="/electricity" element={<ElectricityPage />} />
          <Route path="/info" element={<InfoPage />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App