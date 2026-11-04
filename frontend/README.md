# NewsIntel v3.0.0 — Frontend Cockpit & Tactical HUD

This directory contains the user interface for **NewsIntel v3.0.0**, an enterprise-grade NLP topic classification cockpit designed for sub-millisecond headline inference, explainability inspection, and model telemetry.

---

## Architecture Overview

- **Framework**: Next.js 14.2.35 (App Router, TypeScript, React 18, TailwindCSS).
- **Tactical Cockpit**: The primary mission interface is hosted at `public/hud.html` (and rendered directly via `src/app/page.tsx`), featuring:
  - Three.js WebGL procedural background particle space.
  - GSAP tactical boot decryption animation.
  - 240-degree calibrated confidence arc gauge with Platt sigmoidal posterior display.
  - Fitted-coefficient token-level attribution bars (explainability).
  - 13-class real-time probability rank distributions.
  - Dynamic SQLite prediction audit table with live pagination and CSV export.
  - Live synchronization with `/api/v2/categories` and `/api/v2/benchmarks`.

---

## Security & Sanitization

- **DOM XSS Elimination**: All dynamic user-submitted headlines and backend strings are rendered strictly via `element.textContent` or safe character escaping (`escapeHtml`).
- **Zero Mock Fallback**: The cockpit communicates directly with the FastAPI backend (`http://127.0.0.1:8000`). If the backend is offline or an error occurs, explicit connection error diagnostics are shown. Random mock predictions are strictly prohibited.
- **Environment Isolation**: `.env*` files and local secrets are ignored by `.gitignore`.

---

## Getting Started

### Prerequisites

- Node.js >= 18.17.0 (Recommended: Node.js 20 LTS or 22/24)
- npm >= 9.0.0

### Installation

```bash
# Clean install exactly from package-lock.json
npm ci
```

### Development Mode

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. Ensure the FastAPI backend is running simultaneously on `http://127.0.0.1:8000`.

### Production Build & Verification

```bash
# Compile and optimize production build
npm run build

# Start production server
npm run start
```

---

## Backend Integration

The frontend connects to the following FastAPI endpoints:
- `POST /api/v2/predict`: Primary 13-class headline topic classification with token attributions and Platt posterior probabilities.
- `GET /api/v2/categories`: Metadata for the 13 active news categories, color codes, and seed test headlines.
- `GET /api/v2/benchmarks`: Precomputed benchmark metrics from the held-out test split ($n=2,600$).
- `GET /history`: Persistent SQLite prediction audit history with pagination (`limit`, `offset`).
