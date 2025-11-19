import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function ElectricityPage() {
  return (
    <div className="flex justify-center items-start min-h-[calc(100vh-64px)] p-8 bg-slate-50 dark:bg-slate-950">
      <Card className="w-full max-w-4xl shadow-2xl dark:bg-slate-900 border-yellow-500/50">
        <CardHeader className="bg-yellow-500/10 dark:bg-yellow-500/20 border-b border-yellow-500/30">
          <CardTitle className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
            Predicción de Consumo de Electricidad
          </CardTitle>
        </CardHeader>

        <CardContent className="p-6">
          <p className="text-xl text-slate-700 dark:text-slate-300">
            ¡Estás en la página{" "}
            <span className="font-mono bg-amber-100 dark:bg-amber-900 p-1 rounded">
              /electricity
            </span>
            !
          </p>

          <p className="mt-4 text-slate-500 dark:text-slate-400">
            Aquí irá el formulario para ingresar los datos climáticos y de uso para la predicción.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}