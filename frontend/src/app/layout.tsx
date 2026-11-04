import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NewsIntel | Topic Classification & News Intelligence Platform",
  description: "Scalable 13-Category News Topic Classification Platform with Probability Calibration, Token Explainability & Analytics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#090d16] text-slate-100 selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
