import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom" // <--- 1. Import Navigate
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
          {/* 2. Change the "/" route to use Navigate */}
          <Route path="/" element={<Navigate to="/housing" replace />} />
          
          {/* 3. Keep your other routes */}
          <Route path="/housing" element={<HousingPage />} />
          <Route path="/electricity" element={<ElectricityPage />} />
          <Route path="/info" element={<InfoPage />} />
          
          {/* Optional: Add the modal as a separate route if you still need it, 
              or render it inside HousingPage */}
          {/* <Route path="/prediction" element={<HousePredictionModal />} /> */}
        </Routes>
      </div>
    </Router>
  )
}

export default App