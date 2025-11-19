import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"

export default function HousingPage() {
  return (
    <div className="flex justify-center items-start min-h-[calc(100vh-64px)] p-8 bg-slate-50 dark:bg-slate-950">
      <Card className="w-full max-w-4xl shadow-2xl dark:bg-slate-900 border-blue-500/50">
        <CardHeader className="bg-blue-500/10 dark:bg-blue-500/20 border-b border-blue-500/30">
          <CardTitle className="text-3xl font-bold text-blue-600 dark:text-blue-400">
            Predicción de Precios de Vivienda
          </CardTitle>
        </CardHeader>

        <CardContent className="p-6">
          <p className="text-xl text-slate-700 dark:text-slate-300">
            ¡Estás en la página{" "}
            <span className="font-mono bg-blue-100 dark:bg-blue-900 p-1 rounded">
              /housing
            </span>
            !
          </p>

          <p className="mt-4 text-slate-500 dark:text-slate-400">
            Aquí irá el formulario para predecir el precio de una vivienda en el Reino Unido.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}