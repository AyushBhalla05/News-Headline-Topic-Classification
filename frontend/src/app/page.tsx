"use client";

import React from "react";

export default function Home() {
  return (
    <main className="fixed inset-0 w-screen h-screen overflow-hidden bg-[#03050a]">
      <iframe
        src="/hud.html"
        title="NewsIntel Neural HUD Cockpit"
        className="w-full h-full border-none block"
        style={{ width: "100vw", height: "100vh" }}
        allow="autoplay"
      />
    </main>
  );
}
