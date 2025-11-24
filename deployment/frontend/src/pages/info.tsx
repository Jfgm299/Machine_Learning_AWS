"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Linkedin, User } from "lucide-react";
import { motion } from "framer-motion";

export default function InfoPage() {
  const team = [
    {
      name: "José Fernando Gutiérrez Montero",
      linkedin: "https://www.linkedin.com/in/jf-gutiérrez-montero-963159168/",
      color: "text-blue-600",
    },
    {
      name: "Tymo Verhaegen",
      linkedin: "https://www.linkedin.com/in/tymo-verhaegen/",
      color: "text-blue-600",
    },
    {
      name: "Neila Fekovic",
      linkedin: null,
      color: "text-gray-400",
    },
  ];

  return (
    <div className="min-h-screen w-full flex flex-col items-center py-16 bg-slate-50 dark:bg-slate-900">
      <h1 className="text-4xl font-bold mb-10 text-slate-800 dark:text-white">
        About the Team
      </h1>

      {/* El grid ya se encarga de alinear las filas, pero necesitamos que los hijos se estiren */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl px-6">
        {team.map((member, i) => (
          <motion.div
            key={member.name}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15, duration: 0.5 }}
            className="h-full" // ⭐ CAMBIO 1: Asegura que el contenedor de la animación ocupe toda la altura de la columna
          >
            <Card className="shadow-lg hover:shadow-xl transition rounded-2xl border border-slate-300 dark:border-slate-700 h-full flex flex-col justify-between">
              {/* ⭐ CAMBIO 2 (Arriba): 
                  - h-full: Estira la tarjeta.
                  - flex flex-col: Permite usar flexbox vertical.
                  - justify-between: Pone el Header arriba y el Content (botón) abajo del todo.
              */}
              
              <CardHeader className="flex flex-col items-center">
                <User className="h-16 w-16 text-indigo-500 mb-3" />
                {/* Opcional: Puedes poner un min-h aquí si quieres que los nombres empiecen alineados también, pero justify-between suele bastar */}
                <CardTitle className="text-center text-lg font-semibold text-slate-800 dark:text-slate-100">
                  {member.name}
                </CardTitle>
              </CardHeader>

              <CardContent className="flex justify-center pb-6">
                {member.linkedin ? (
                  <a
                    href={member.linkedin}
                    target="_blank"
                    rel="noopener noreferrer" // Buena práctica de seguridad para target="_blank"
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full transition font-medium"
                  >
                    <Linkedin className="h-5 w-5" />
                    LinkedIn
                  </a>
                ) : (
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-300 text-gray-600 rounded-full cursor-not-allowed font-medium">
                    <Linkedin className="h-5 w-5" />
                    No LinkedIn
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
}