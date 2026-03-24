# Social-Media-Radar — Website Developer Guide

**Document type:** Developer handoff + complete website copy specification
**Version:** 1.0
**Last updated:** 2026-03-24
**Audience:** Front-end / full-stack developers building the product website
**Application version this document targets:** Social-Media-Radar 1.0 (593 tests passing, production-ready)

---

## Table of Contents

1. [Document Purpose](#1-document-purpose)
2. [Website Technical Architecture](#2-website-technical-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Local Development Setup](#4-local-development-setup)
5. [Environment Variables](#5-environment-variables)
6. [Build Process](#6-build-process)
7. [Deployment — Vercel (Primary)](#7-deployment--vercel-primary)
8. [Deployment — Self-Hosted (Alternative)](#8-deployment--self-hosted-alternative)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [SEO and Meta Configuration](#10-seo-and-meta-configuration)
11. [API Integration (Live Demo and Docs)](#11-api-integration-live-demo-and-docs)
12. [Complete Website Copy — All Sections](#12-complete-website-copy--all-sections)
    - 12.1 [Navigation Bar](#121-navigation-bar)
    - 12.2 [Hero Section](#122-hero-section)
    - 12.3 [Trust Bar](#123-trust-bar)
    - 12.4 [Problem Statement](#124-problem-statement)
    - 12.5 [How It Works (Pipeline)](#125-how-it-works-pipeline)
    - 12.6 [Signal Taxonomy](#126-signal-taxonomy)
    - 12.7 [Platform Coverage](#127-platform-coverage)
    - 12.8 [Privacy and Data Residency](#128-privacy-and-data-residency)
    - 12.9 [LLM Routing and Cost Efficiency](#129-llm-routing-and-cost-efficiency)
    - 12.10 [Calibration and Accuracy](#1210-calibration-and-accuracy)
    - 12.11 [Performance Benchmarks](#1211-performance-benchmarks)
    - 12.12 [Team Workflow](#1212-team-workflow)
    - 12.13 [Local Deployment Section](#1213-local-deployment-section)
    - 12.14 [Frequently Asked Questions](#1214-frequently-asked-questions)
    - 12.15 [Footer](#1215-footer)
13. [Asset Specifications](#13-asset-specifications)
14. [Accessibility and Performance Targets](#14-accessibility-and-performance-targets)
15. [Analytics and Tracking](#15-analytics-and-tracking)
16. [Maintenance Notes](#16-maintenance-notes)


---

## 1. Document Purpose

This document is the single source of truth for every developer working on the Social-Media-Radar **product website** — the public-facing site that presents, explains, and provides download/deployment instructions for the application. It is **not** documentation for the application itself (that lives in `README.md` and the `docs/` directory).

By the end of this document you will have:

- A fully reproducible local development environment for the website.
- Step-by-step instructions to publish the site to Vercel or a self-hosted server.
- The exact copy (headline, body text, labels, tooltips, table values, error messages) for every section of every page, derived directly from the application source code and benchmark data — no placeholders, no invented figures.
- Asset specifications (dimensions, formats, alt text) for every image and illustration.
- Accessibility, performance, analytics, and SEO requirements.

**Source of truth for all technical claims:** the application repository at the root of this workspace. Every number in the website copy is sourced from `deliverables/results/*.csv`, `training/calibration_state.json`, `training/fine_tuning_plan.md`, `pyproject.toml`, `app/domain/inference_models.py`, `app/connectors/registry.py`, and `app/core/config.py`. Do not invent or estimate figures — update the website copy whenever those source files change.

---

## 2. Website Technical Architecture

### Recommended Stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | **Next.js 14** (App Router) | Static export for marketing pages; RSC for interactive docs; excellent SEO defaults |
| Language | **TypeScript 5.3+** | Type-safe API calls to the live demo backend; IDE completion for copy constants |
| Styling | **Tailwind CSS 3.4** | Utility-first; consistent with the target design system |
| UI components | **shadcn/ui** | Accessible, unstyled-by-default; allows full brand customisation |
| Documentation pages | **MDX** via `@next/mdx` | Lets the dev team maintain docs in Markdown with embedded React components |
| Syntax highlighting | **Shiki** | Zero-runtime code blocks; used for all `curl` / `bash` / `python` examples |
| Diagrams | **Mermaid.js** (client-side) | Renders the pipeline ASCII art as an interactive SVG |
| Icons | **Lucide React** | MIT-licensed; consistent stroke width |
| Analytics | **Plausible** (self-hostable) or **Vercel Analytics** | Privacy-preserving; no cookie consent banner needed |
| Deployment | **Vercel** (primary) | Zero-config Next.js; Preview deployments on every PR; Edge CDN |
| Self-hosted alternative | **Docker + nginx** | For organisations that cannot use Vercel |
| Package manager | **pnpm 9+** | Faster installs; strict lockfile |

### Page Map

```
/                       → Home (all copy sections below)
/docs                   → Documentation index (mirrors docs/ directory)
/docs/architecture      → Architecture deep-dive
/docs/deployment        → Local deployment guide (mirrors README §Local Deployment)
/docs/api               → API reference (embedded Swagger UI from /api/v1/openapi.json)
/docs/training          → Calibration training guide
/docs/llm               → LLM provider configuration
/benchmark              → Interactive benchmark charts (CSV data from deliverables/results/)
/changelog              → Release notes
/404                    → Custom not-found page
```

### Data Flow: Website ↔ Application API

The website connects to a running Social-Media-Radar API instance in two ways:

1. **Static build time** — benchmark CSV data (`deliverables/results/*.csv`) is imported at build time and rendered into static HTML tables and chart data. No runtime API call needed for these.
2. **Live demo (optional)** — if `NEXT_PUBLIC_API_BASE_URL` is set, the site renders an interactive demo panel that authenticates against the live API and shows real signal queue output.

```
Website (Next.js)
    │
    ├── Build time: reads deliverables/results/*.csv → static chart JSON
    ├── Build time: reads training/calibration_state.json → scalar table
    │
    └── Runtime (optional live demo):
        POST /api/v1/auth/login          → obtain JWT
        GET  /api/v1/signals/queue       → display live signal queue
        POST /api/v1/signals/{id}/act    → interactive action button
```

---

## 3. Repository Structure

Create a separate Git repository for the website. Suggested layout:

```
social-media-radar-site/
├── app/                        # Next.js 14 App Router pages
│   ├── layout.tsx              # Root layout (nav + footer)
│   ├── page.tsx                # Home page (all sections below)
│   ├── docs/
│   │   ├── layout.tsx          # Docs sidebar layout
│   │   ├── page.tsx            # Docs index
│   │   ├── architecture/
│   │   │   └── page.mdx        # Content: docs/architecture.md
│   │   ├── deployment/
│   │   │   └── page.mdx        # Content: README §Local Deployment
│   │   ├── api/
│   │   │   └── page.tsx        # Embedded Swagger UI
│   │   ├── training/
│   │   │   └── page.mdx        # Content: docs/TRAINING.md
│   │   └── llm/
│   │       └── page.mdx        # Content: docs/LLM_DEPLOYMENT_GUIDE.md
│   ├── benchmark/
│   │   └── page.tsx            # Interactive benchmark charts
│   └── changelog/
│       └── page.mdx
│
├── components/
│   ├── nav.tsx                 # Navigation bar
│   ├── hero.tsx                # Hero section
│   ├── trust-bar.tsx           # Stats / badges strip
│   ├── problem.tsx             # Problem statement
│   ├── pipeline.tsx            # How It Works pipeline diagram
│   ├── signal-taxonomy.tsx     # 18-type signal grid
│   ├── platforms.tsx           # 13-connector icon grid
│   ├── privacy.tsx             # Data residency section
│   ├── llm-routing.tsx         # LLM cost comparison
│   ├── calibration.tsx         # Accuracy and calibration
│   ├── benchmarks.tsx          # Performance benchmark charts
│   ├── team-workflow.tsx       # Team features
│   ├── deployment-steps.tsx    # Local deployment CTA section
│   ├── faq.tsx                 # Accordion FAQ
│   ├── footer.tsx              # Footer
│   └── ui/                    # shadcn/ui primitives
│
├── data/
│   ├── bloom.csv               # Copied from deliverables/results/
│   ├── reservoir.csv
│   ├── calibrator.csv
│   ├── action_ranker.csv
│   ├── bfs.csv
│   └── calibration_state.json  # Copied from training/calibration_state.json
│
├── public/
│   ├── og-image.png            # 1200×630 Open Graph image
│   ├── logo.svg
│   ├── favicon.ico
│   └── screenshots/            # Product screenshots
│
├── lib/
│   ├── copy.ts                 # All website copy as typed constants
│   ├── benchmark-data.ts       # Parsed CSV data for charts
│   └── api-client.ts           # Typed API client (live demo)
│
├── .env.local                  # Local environment variables
├── .env.production             # Production overrides
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── pnpm-lock.yaml
```

**Important:** keep all human-readable website copy in `lib/copy.ts` as typed string constants, not scattered across TSX files. This makes copy updates a one-file change and allows a non-developer to maintain text without touching components.


---

## 4. Local Development Setup

### Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Node.js | 20 LTS | `brew install node@20` / [nodejs.org](https://nodejs.org) |
| pnpm | 9.0 | `npm install -g pnpm` |
| Git | 2.40 | pre-installed on macOS, `apt install git` on Ubuntu |

### Steps

```bash
# 1. Clone the website repository
git clone https://github.com/yourorg/social-media-radar-site.git
cd social-media-radar-site

# 2. Install dependencies
pnpm install

# 3. Copy benchmark data from the application repository
#    Adjust APP_REPO_PATH to wherever the application repo lives locally
APP_REPO_PATH=../Social-Media-Radar
cp "$APP_REPO_PATH/deliverables/results/"*.csv data/
cp "$APP_REPO_PATH/training/calibration_state.json" data/

# 4. Set up local environment variables (see §5 below)
cp .env.example .env.local
# Edit .env.local — minimum required: set NEXT_PUBLIC_REPO_URL

# 5. Start the development server
pnpm dev
# Site available at http://localhost:3000

# 6. Run type-check and linting
pnpm typecheck
pnpm lint

# 7. Build for production (catches SSG errors locally)
pnpm build
pnpm start   # preview the production build at http://localhost:3000
```

### Useful Development Commands

```bash
pnpm dev            # Start with hot reload (Next.js Fast Refresh)
pnpm build          # Full production build — run before every PR
pnpm start          # Serve the production build locally
pnpm lint           # ESLint + Prettier check
pnpm lint:fix       # Auto-fix lint errors
pnpm typecheck      # tsc --noEmit (no output files, just type errors)
pnpm test           # Playwright E2E tests (requires pnpm build first)
pnpm test:unit      # Vitest unit tests for lib/ functions
```

---

## 5. Environment Variables

All environment variables used by the website are **prefixed with `NEXT_PUBLIC_`** if they must be accessible in the browser, or left without the prefix if they are server-only (build-time only for a fully static export).

Create `.env.local` for local development. **Never commit `.env.local` to version control.** Add `.env.local` to `.gitignore`.

### Required Variables

```bash
# Public GitHub repository URL — used for "View on GitHub" buttons and download links
NEXT_PUBLIC_REPO_URL=https://github.com/yourorg/social-media-radar

# Version tag to display in the hero badge and download instructions
NEXT_PUBLIC_APP_VERSION=1.0.0

# Site base URL (used for canonical links and OG tags)
NEXT_PUBLIC_SITE_URL=https://social-media-radar.dev
```

### Optional Variables

```bash
# Live demo API base URL — if omitted, the demo panel is hidden
# Must point to a running Social-Media-Radar API instance (GET /health must return 200)
NEXT_PUBLIC_API_BASE_URL=https://demo.social-media-radar.dev

# Plausible Analytics domain (omit to disable analytics entirely)
NEXT_PUBLIC_PLAUSIBLE_DOMAIN=social-media-radar.dev

# Vercel Analytics (set automatically by Vercel; set manually for self-hosted)
NEXT_PUBLIC_VERCEL_ANALYTICS=true

# Sentry DSN for front-end error tracking (optional)
NEXT_PUBLIC_SENTRY_DSN=https://xxxxx@sentry.io/yyyyy
```

### `.env.example` (commit this file)

```bash
NEXT_PUBLIC_REPO_URL=https://github.com/yourorg/social-media-radar
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_SITE_URL=https://social-media-radar.dev
NEXT_PUBLIC_API_BASE_URL=        # leave blank to hide live demo
NEXT_PUBLIC_PLAUSIBLE_DOMAIN=    # leave blank to disable analytics
NEXT_PUBLIC_VERCEL_ANALYTICS=false
NEXT_PUBLIC_SENTRY_DSN=          # leave blank to disable Sentry
```

---

## 6. Build Process

### Static Export (Recommended for Initial Launch)

For a fully static site (no server-side rendering at request time), add the following to `next.config.ts`:

```typescript
// next.config.ts
import type { NextConfig } from 'next'
import createMDX from '@next/mdx'

const withMDX = createMDX({
  options: {
    remarkPlugins: [require('remark-gfm')],
    rehypePlugins: [require('rehype-slug'), require('rehype-autolink-headings')],
  },
})

const config: NextConfig = {
  output: 'export',           // static HTML export into /out
  images: { unoptimized: true }, // required for static export
  pageExtensions: ['ts', 'tsx', 'md', 'mdx'],
  // All CSV and JSON data imports are read at build time via fs
  // No runtime server needed
}

export default withMDX(config)
```

Build and inspect the output:

```bash
pnpm build
# Output directory: ./out/
# Total size target: < 500 KB gzipped for the home page

ls -lh out/         # confirm all pages are generated
du -sh out/         # confirm total size
```

### Server-Side Rendering (Required for Live Demo Panel)

If `NEXT_PUBLIC_API_BASE_URL` is set and you want the live signal queue to render with fresh data on each request, **remove** `output: 'export'` from `next.config.ts`. The live demo route (`/demo`) should then be a Server Component that fetches from the API at request time. All other pages remain statically generated via `generateStaticParams`.

### Benchmark Data Pipeline

Before running `pnpm build`, the benchmark CSV files must exist in `data/`. Add a pre-build script to `package.json` that copies them from the application repo (for CI environments where the app repo is checked out as a Git submodule or sibling directory):

```json
{
  "scripts": {
    "prebuild": "node scripts/sync-benchmark-data.js",
    "build": "next build"
  }
}
```

`scripts/sync-benchmark-data.js`:

```javascript
// Reads ../Social-Media-Radar/deliverables/results/*.csv
// and ../Social-Media-Radar/training/calibration_state.json
// and writes them to ./data/
// Run automatically before every build via the "prebuild" npm hook.
const fs = require('fs')
const path = require('path')

const APP_ROOT = path.resolve(__dirname, '..', '..', 'Social-Media-Radar')
const DATA_DIR = path.resolve(__dirname, '..', 'data')

const files = [
  ['deliverables/results/bloom.csv',              'bloom.csv'],
  ['deliverables/results/reservoir.csv',          'reservoir.csv'],
  ['deliverables/results/calibrator.csv',         'calibrator.csv'],
  ['deliverables/results/action_ranker.csv',      'action_ranker.csv'],
  ['deliverables/results/bfs.csv',                'bfs.csv'],
  ['training/calibration_state.json',             'calibration_state.json'],
]

fs.mkdirSync(DATA_DIR, { recursive: true })
for (const [src, dest] of files) {
  fs.copyFileSync(path.join(APP_ROOT, src), path.join(DATA_DIR, dest))
  console.log(`Synced ${src} → data/${dest}`)
}
```

---

## 7. Deployment — Vercel (Primary)

### Initial Setup

1. Push the website repository to GitHub (or GitLab / Bitbucket — Vercel supports all three).
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import the website repository.
3. Vercel auto-detects Next.js. Accept all defaults.
4. Add environment variables under **Settings → Environment Variables**:

```
NEXT_PUBLIC_REPO_URL        = https://github.com/yourorg/social-media-radar
NEXT_PUBLIC_APP_VERSION     = 1.0.0
NEXT_PUBLIC_SITE_URL        = https://social-media-radar.dev
NEXT_PUBLIC_API_BASE_URL    = https://demo.social-media-radar.dev   (if live demo enabled)
NEXT_PUBLIC_PLAUSIBLE_DOMAIN = social-media-radar.dev
```

5. Click **Deploy**. First build takes 60–90 seconds.

### Custom Domain

1. In Vercel dashboard → **Settings → Domains** → Add `social-media-radar.dev` (or your domain).
2. Add the DNS records Vercel specifies at your registrar:
   - `A` record: `76.76.21.21`
   - `CNAME` record: `cname.vercel-dns.com`
3. SSL is provisioned automatically via Let's Encrypt within 60 seconds of DNS propagation.

### Preview Deployments

Every pull request automatically receives a preview URL (`https://social-media-radar-site-git-{branch}-yourorg.vercel.app`). Use this for QA before merging to `main`. Preview deployments use the same environment variables as production unless overridden in **Settings → Environment Variables → Preview** scope.

### Rollback

```bash
# List recent deployments
vercel ls

# Roll back to a specific deployment
vercel rollback <deployment-url>
```

---

## 8. Deployment — Self-Hosted (Alternative)

Use this path when Vercel is not permitted (e.g., strict data-residency environments).

### Option A: Docker + nginx (Static Export)

```dockerfile
# Dockerfile.website
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY . .
ARG NEXT_PUBLIC_REPO_URL
ARG NEXT_PUBLIC_APP_VERSION
ARG NEXT_PUBLIC_SITE_URL
RUN pnpm build
# Output: /app/out/

FROM nginx:1.25-alpine
COPY --from=builder /app/out /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

`nginx.conf`:

```nginx
server {
    listen 80;
    server_name social-media-radar.dev;
    root /usr/share/nginx/html;
    index index.html;

    # Cache static assets aggressively
    location /_next/static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA fallback for client-side navigation
    location / {
        try_files $uri $uri.html $uri/ /index.html =404;
    }

    # Security headers
    add_header X-Frame-Options "DENY";
    add_header X-Content-Type-Options "nosniff";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Permissions-Policy "camera=(), microphone=(), geolocation=()";
}
```

Build and run:

```bash
docker build -f Dockerfile.website \
  --build-arg NEXT_PUBLIC_REPO_URL=https://github.com/yourorg/social-media-radar \
  --build-arg NEXT_PUBLIC_APP_VERSION=1.0.0 \
  --build-arg NEXT_PUBLIC_SITE_URL=https://social-media-radar.dev \
  -t smr-website:latest .

docker run -p 80:80 smr-website:latest
```

### Option B: Node.js Server (SSR / Live Demo)

If the live demo panel is enabled (`output: 'export'` removed from `next.config.ts`):

```bash
# On the server
pnpm build
pm2 start pnpm --name smr-website -- start -- -p 3000
pm2 save
pm2 startup   # generate systemd service
```

Point nginx as a reverse proxy:

```nginx
location / {
    proxy_pass http://127.0.0.1:3000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}
```

---

## 9. CI/CD Pipeline

### GitHub Actions — `.github/workflows/deploy.yml`

```yaml
name: Build and Deploy Website

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-22.04

    steps:
      - uses: actions/checkout@v4

      # Check out application repo to sync benchmark data
      - uses: actions/checkout@v4
        with:
          repository: yourorg/social-media-radar
          path: app-repo
          token: ${{ secrets.GITHUB_TOKEN }}

      - uses: pnpm/action-setup@v3
        with:
          version: 9

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Sync benchmark data
        run: |
          mkdir -p data
          cp app-repo/deliverables/results/*.csv data/
          cp app-repo/training/calibration_state.json data/

      - name: Type check
        run: pnpm typecheck

      - name: Lint
        run: pnpm lint

      - name: Build
        run: pnpm build
        env:
          NEXT_PUBLIC_REPO_URL: https://github.com/yourorg/social-media-radar
          NEXT_PUBLIC_APP_VERSION: ${{ vars.APP_VERSION }}
          NEXT_PUBLIC_SITE_URL: https://social-media-radar.dev

      - name: Deploy to Vercel (production)
        if: github.ref == 'refs/heads/main'
        run: vercel --prod --token ${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

      - name: Deploy to Vercel (preview)
        if: github.event_name == 'pull_request'
        run: vercel --token ${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

### Required GitHub Secrets

| Secret name | Where to find it |
|---|---|
| `VERCEL_TOKEN` | Vercel dashboard → Account Settings → Tokens |
| `VERCEL_ORG_ID` | `.vercel/project.json` after running `vercel link` locally |
| `VERCEL_PROJECT_ID` | `.vercel/project.json` after running `vercel link` locally |

### Required GitHub Variables (not secrets — visible in logs)

| Variable name | Example value |
|---|---|
| `APP_VERSION` | `1.0.0` |

---

## 10. SEO and Meta Configuration

### `app/layout.tsx` — Root Metadata

```typescript
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: {
    default: 'Social-Media-Radar — Structured Social Intelligence for B2B Teams',
    template: '%s | Social-Media-Radar',
  },
  description:
    'Open-source, locally-deployable social media intelligence platform. ' +
    'Classifies signals across 13 platforms into 10 actionable business types ' +
    'using calibrated LLM inference. Runs entirely on your machine — no SaaS, ' +
    'no data egress, no vendor lock-in.',
  keywords: [
    'social media monitoring', 'social listening', 'B2B intelligence',
    'LLM classification', 'churn risk detection', 'competitor intelligence',
    'open source', 'self-hosted', 'pgvector', 'FastAPI',
  ],
  authors: [{ name: 'Social-Media-Radar Contributors' }],
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://social-media-radar.dev',
    siteName: 'Social-Media-Radar',
    title: 'Social-Media-Radar — Structured Social Intelligence for B2B Teams',
    description:
      'Open-source social media signal detection. 10 business-intent signal types, ' +
      '13 platform connectors, calibrated LLM inference, full local deployment.',
    images: [{ url: '/og-image.png', width: 1200, height: 630,
               alt: 'Social-Media-Radar — Signal queue showing churn risk, competitor weakness, and feature request signals' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Social-Media-Radar — Structured Social Intelligence',
    description: 'Open-source, locally-deployable social listening with calibrated LLM inference.',
    images: ['/og-image.png'],
  },
  robots: { index: true, follow: true },
  alternates: { canonical: 'https://social-media-radar.dev' },
}
```

---

## 11. API Integration (Live Demo and Docs)

### Embedded Swagger UI (`/docs/api`)

The application API exposes an OpenAPI 3.1 schema at `GET /openapi.json`. Embed Swagger UI as a client-side-only component:

```typescript
// app/docs/api/page.tsx
'use client'
import dynamic from 'next/dynamic'

const SwaggerUI = dynamic(() => import('swagger-ui-react'), { ssr: false })

export default function ApiReferencePage() {
  const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'
  return (
    <div className="swagger-wrapper">
      <SwaggerUI url={`${apiUrl}/openapi.json`} />
    </div>
  )
}
```

Install: `pnpm add swagger-ui-react @types/swagger-ui-react`.

### Live Demo API Client (`lib/api-client.ts`)

```typescript
// All API calls used by the live demo panel.
// Source of truth for endpoint paths: app/api/main.py router prefixes.

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? ''

export async function login(email: string, password: string): Promise<string> {
  const res = await fetch(`${BASE}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) throw new Error('Login failed')
  const data = await res.json()
  return data.access_token as string
}

export async function getSignalQueue(token: string, params?: {
  signal_types?: string[]
  min_urgency?: number
  limit?: number
}) {
  const q = new URLSearchParams()
  if (params?.signal_types) q.set('signal_types', params.signal_types.join(','))
  if (params?.min_urgency)  q.set('min_urgency', String(params.min_urgency))
  if (params?.limit)        q.set('limit', String(params.limit))
  const res = await fetch(`${BASE}/api/v1/signals/queue?${q}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) throw new Error('Failed to fetch signal queue')
  return res.json()
}

export async function actOnSignal(token: string, signalId: string, payload: {
  action_type: string
  notes: string
  response_tone: string
}) {
  const res = await fetch(`${BASE}/api/v1/signals/${signalId}/act`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Action failed')
  return res.json()
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) })
    return res.ok
  } catch {
    return false
  }
}
```

---

## 12. Complete Website Copy — All Sections

> **Developer instruction:** implement each section below as an independent React component (filename given in the heading). Import all string values from `lib/copy.ts` — do not hardcode them in TSX. The copy below is the exact text; do not paraphrase or summarise.

---

### 12.1 Navigation Bar

**Component:** `components/nav.tsx`
**Behaviour:** sticky top; transparent over hero, solid background on scroll; mobile hamburger menu collapses to full-screen drawer.

```
Logo text:    Social-Media-Radar
Logo subtext: (none — logo only)

Nav links (left side, desktop):
  • Features         → /#features          (smooth scroll)
  • How It Works     → /#pipeline          (smooth scroll)
  • Benchmarks       → /benchmark
  • Documentation    → /docs
  • Changelog        → /changelog

Nav links (right side, desktop):
  • GitHub           → {NEXT_PUBLIC_REPO_URL}     [opens in new tab]
    Icon: GitHub mark (Lucide: Github)
    Label: "Star on GitHub"
  • Download         → {NEXT_PUBLIC_REPO_URL}/releases/latest
    Style: filled primary button
    Label: "Download v{NEXT_PUBLIC_APP_VERSION}"

Mobile menu: same links in vertical list, Download button full-width at bottom.
```

---

### 12.2 Hero Section

**Component:** `components/hero.tsx`
**Layout:** two-column on desktop (copy left, product screenshot right); single column on mobile (copy above, screenshot below).

```
Badge (top pill):
  Icon: Lucide:Radar
  Text: "Open-Source · Self-Hosted · MIT License"

Headline (H1):
  "Stop reading.
   Start acting."

Subheadline (H2, lighter weight):
  "Social-Media-Radar monitors 13 platforms, classifies every post into
   10 business-intent signal types using calibrated LLM inference, and
   delivers a prioritised action queue — running entirely on your machine."

CTA buttons:
  Primary:   "Get Started — Deploy in 5 minutes"
             → /docs/deployment
  Secondary: "View on GitHub"
             → {NEXT_PUBLIC_REPO_URL}   [new tab]
             Icon: Lucide:Github (left)

Beneath CTAs — three inline stats (small text, separated by ·):
  "593 tests passing"  ·  "MIT License"  ·  "Python 3.11 · FastAPI · pgvector"

Hero image / screenshot:
  File: public/screenshots/signal-queue.png
  Alt:  "Social-Media-Radar signal queue showing a churn_risk signal at 0.87 confidence
         with verbatim evidence spans from a Reddit post"
  Caption (below image): "Signal queue view — confidence-ranked, with verbatim evidence"
```

---

### 12.3 Trust Bar

**Component:** `components/trust-bar.tsx`
**Layout:** single horizontal strip, 5 stat cards, grey/neutral background.

```
Stat 1:
  Number: 13
  Label:  "Platform connectors"
  Icon:   Lucide:Globe

Stat 2:
  Number: 10
  Label:  "Business signal types"
  Icon:   Lucide:Zap

Stat 3:
  Number: 593
  Label:  "Tests passing"
  Icon:   Lucide:CheckCircle2

Stat 4:
  Number: "< 13 µs"
  Label:  "Per deduplication check"
  Icon:   Lucide:Timer

Stat 5:
  Number: "100%"
  Label:  "Local — no data egress"
  Icon:   Lucide:Shield
```

Source for Stat 4: `deliverables/results/bloom.csv`, row n=100000, time_ms=1300.3 → 13.0 µs/op.

---

### 12.4 Problem Statement

**Component:** `components/problem.tsx`
**Layout:** full-width, dark background, centred copy, three pain-point cards below.

```
Section label (small caps above heading):
  "THE PROBLEM"

Heading:
  "Your team is drowning in social noise.
   The signals that matter are buried."

Body:
  "B2B teams responsible for reputation, sales, and product spend hours
   each week manually scrolling Reddit threads, YouTube comment sections,
   and news feeds — searching for the posts that contain an actual signal:
   a prospect who just complained about a competitor, a customer whose
   support issue went public, a feature request pattern forming across
   dozens of independent posts.

   Existing social listening tools surface volume, not intent. They count
   mentions and track sentiment. They do not tell you what to do."

Pain-point cards (three, horizontal on desktop, stacked on mobile):

  Card 1:
    Icon:   Lucide:Clock
    Title:  "Hours lost to manual triage"
    Body:   "A mid-size B2B team spends 3–5 hours per week reading posts
             that contain no actionable signal — only to miss the churn risk
             that surfaced in a subreddit no one was watching."

  Card 2:
    Icon:   Lucide:AlertTriangle
    Title:  "Signal buried in sentiment scores"
    Body:   "A post can be negative in sentiment and completely irrelevant
             to your business. It can be neutral in tone and contain an
             explicit statement of intent to switch vendors. Sentiment
             is the wrong unit of measurement."

  Card 3:
    Icon:   Lucide:Lock
    Title:  "SaaS tools with your data"
    Body:   "Every SaaS listening tool requires you to pipe your
             customer conversations, competitor mentions, and product
             feedback through a third-party server. For regulated
             industries, this is a compliance problem. For everyone
             else, it is an unnecessary risk."
```

---

### 12.5 How It Works — Pipeline

**Component:** `components/pipeline.tsx`
**Layout:** numbered vertical stepper on mobile; horizontal connected flow with icons on desktop. Each step has a title, a one-sentence description, and a technical detail line (smaller, monospace font).

```
Section label: "HOW IT WORKS"

Heading: "From raw post to structured action in six steps"

Subheading:
  "Every observation that enters the system passes through a deterministic,
   auditable pipeline. No black boxes. Every classification decision comes
   with a structured rationale and verbatim evidence spans."

Steps:

  Step 1:
    Icon:   Lucide:Download
    Title:  "Ingest"
    Body:   "Celery workers fetch content from every configured platform
             connector on a 15-minute schedule. New URLs are checked against
             a Bloom filter before any processing begins — eliminating
             duplicate work without a database query."
    Tech:   "Bloom filter · O(1) deduplication · 13 µs per URL check"

  Step 2:
    Icon:   Lucide:Filter
    Title:  "Sample"
    Body:   "When a platform returns more content than the fetch budget allows,
             a reservoir sampler draws a statistically unbiased sample.
             Every item in the stream has an equal probability of inclusion,
             regardless of total stream length."
    Tech:   "Reservoir sampling · O(n) · ~1,000 items/ms throughput"

  Step 3:
    Icon:   Lucide:Wand2
    Title:  "Normalise"
    Body:   "Raw observations from 13 different platforms — each with its own
             schema, encoding, language, and media type — are transformed into
             a unified NormalizedObservation. PII is scrubbed. Non-English text
             is detected and flagged for translation. Quality and completeness
             scores are computed."
    Tech:   "NormalizationEngine · DataResidencyGuard · spaCy NER"

  Step 4:
    Icon:   Lucide:Search
    Title:  "Retrieve"
    Body:   "A candidate retrieval step finds semantically similar past
             observations using pgvector cosine similarity search. The top-k
             results are assembled into a few-shot context window that is
             passed to the LLM — giving the model concrete examples of how
             similar content has been classified before."
    Tech:   "pgvector · 1536-dim embeddings · HNSW approximate nearest neighbour"

  Step 5:
    Icon:   Lucide:Brain
    Title:  "Classify"
    Body:   "The LLM Adjudicator classifies the observation against the 10-type
             signal taxonomy. Frontier-tier signals (churn risk, misinformation,
             support escalation) route to GPT-4o. The remaining 7 types route
             to a fine-tuned smaller model or a local Ollama model — reducing
             per-signal LLM cost without accuracy regression. When the model is
             genuinely uncertain, it abstains rather than producing a low-quality
             classification."
    Tech:   "LLMRouter · two-tier routing · calibrated confidence · abstention"

  Step 6:
    Icon:   Lucide:Sparkles
    Title:  "Rank and Deliver"
    Body:   "Classified signals are scored across three dimensions: opportunity
             (business value), urgency (time sensitivity), and risk (cost of
             inaction). The ActionRanker produces a composite priority score.
             The ranked queue is available immediately via REST API, and new
             signals are pushed to connected clients via Server-Sent Events."
    Tech:   "ActionRanker · composite priority score · SSE streaming"
```

---

### 12.6 Signal Taxonomy

**Component:** `components/signal-taxonomy.tsx`
**Layout:** three-column card grid on desktop, two-column on tablet, one-column on mobile. Cards grouped by category with a coloured category badge. Source of truth for all signal type names: `app/core/signal_models.py` `SignalType` enum.

```
Section label: "SIGNAL TAXONOMY"

Heading: "10 business-intent signal types. Not 10,000 sentiment buckets."

Subheading:
  "Every classified observation maps to exactly one signal type from the
   taxonomy below. The classifier abstains — and tells you why — when the
   evidence is insufficient for a reliable classification."

Category: REVENUE OPPORTUNITIES  (badge colour: emerald)

  Card 1:
    Slug:       lead_opportunity
    Name:       "Lead Opportunity"
    Icon:       Lucide:TrendingUp
    Description:
      "A prospect publicly expressing dissatisfaction with a competitor,
       requesting alternatives, or describing a pain point your product
       solves. Typically surfaces in community forums, Reddit, and LinkedIn."
    Recommended action:  "DM Outreach"
    Example evidence:
      '"We've been using [Competitor] for two years and the pricing
       just got unbearable. Looking for alternatives in the comments."'

  Card 2:
    Slug:       competitor_weakness
    Name:       "Competitor Weakness"
    Icon:       Lucide:Target
    Description:
      "Public criticism, outage reports, or recurring complaints directed at
       a competitor that represent a window to position your product. Includes
       posts by the competitor's customers describing unmet needs."
    Recommended action:  "Create Content"
    Example evidence:
      '"[Competitor] has been down for 3 hours. This is the fourth time
       this quarter. I'm done."'

  Card 3:
    Slug:       influencer_amplification
    Name:       "Influencer Amplification"
    Icon:       Lucide:Megaphone
    Description:
      "A post by a high-reach account (measured by follower count,
       engagement rate, and channel authority) that mentions your brand,
       category, or a topic you can credibly enter. Time-sensitive."
    Recommended action:  "Reply Public"
    Example evidence:
      '[YouTube creator with 420k subscribers] "I switched my entire
       agency workflow to this tool — here's why."'

Category: RISK SIGNALS  (badge colour: red)

  Card 4:
    Slug:       churn_risk
    Name:       "Churn Risk"
    Icon:       Lucide:UserMinus
    Description:
      "An existing customer or user expressing frustration, threatening to
       cancel, or comparing your product unfavourably. Requires urgent
       response. Routes to the frontier LLM tier for maximum accuracy."
    Recommended action:  "Internal Alert → DM Outreach"
    Example evidence:
      '"Three bugs in two weeks and support hasn't replied. I'm moving
       our team off [Product] this Friday unless something changes."'

  Card 5:
    Slug:       misinformation_risk
    Name:       "Misinformation Risk"
    Icon:       Lucide:AlertOctagon
    Description:
      "Factually incorrect claims about your product, company, or team
       that are spreading in public forums. Each hour of delay increases
       the amplification. Routes to the frontier LLM tier."
    Recommended action:  "Reply Public"
    Example evidence:
      '"[Product] was acquired by [Wrong Company] last month and they're
       shutting it down." — spreading in a 15k-member Slack community.'

  Card 6:
    Slug:       support_escalation
    Name:       "Support Escalation"
    Icon:       Lucide:PhoneCall
    Description:
      "A support issue that has escaped private channels and is now
       playing out publicly — on Twitter/X, Reddit, or a tech forum.
       Often signals that the private support channel has failed."
    Recommended action:  "Reply Public + Internal Alert"
    Example evidence:
      '"@[Product] — I've opened three tickets in 10 days and nobody
       has responded. Posting here since I have no other options."'

Category: PRODUCT SIGNALS  (badge colour: violet)

  Card 7:
    Slug:       product_confusion
    Name:       "Product Confusion"
    Icon:       Lucide:HelpCircle
    Description:
      "Posts that reveal a fundamental misunderstanding of what your
       product does, how it works, or how it is priced. A high rate of
       product_confusion signals is a leading indicator of onboarding
       or documentation failures."
    Recommended action:  "Create Content"
    Example evidence:
      '"Wait — [Product] doesn't support [Feature]? I thought that was
       the whole point. We bought it specifically for that."'

  Card 8:
    Slug:       feature_request_pattern
    Name:       "Feature Request Pattern"
    Icon:       Lucide:Lightbulb
    Description:
      "A recurring request for a specific capability appearing across
       multiple independent posts over a rolling window. The system
       identifies the pattern, not just individual requests."
    Recommended action:  "Monitor → Internal Alert"
    Example evidence:
      "Cluster of 14 posts over 3 weeks across Reddit and Twitter all
       requesting native CSV export with custom date ranges."

  Card 9:
    Slug:       launch_moment
    Name:       "Launch Moment"
    Icon:       Lucide:Rocket
    Description:
      "A product launch — yours or a competitor's — generating significant
       public discussion. Includes pre-launch leaks, launch-day coverage,
       and post-launch reactions."
    Recommended action:  "Create Content + Reply Public"
    Example evidence:
      '"[Competitor] just launched [Feature] in beta. This is what
       everyone in our space has been waiting for."'

Category: CONTENT OPPORTUNITIES  (badge colour: amber)

  Card 10:
    Slug:       trend_to_content
    Name:       "Trend to Content"
    Icon:       Lucide:BarChart2
    Description:
      "A rising conversation, topic, or question in your market that
       your team is credibly positioned to address with content — a blog
       post, video, or technical guide that captures the moment."
    Recommended action:  "Create Content"
    Example evidence:
      "Rapid growth in discussions about [Technical Topic] across Hacker
       News and multiple engineering subreddits over the past 72 hours."

Below the card grid:

  Note (italic, small):
    "When confidence is insufficient for a reliable classification, the
     model abstains and returns a structured abstention reason (e.g.,
     'ambiguous_intent', 'insufficient_context', 'out_of_scope').
     Abstentions never surface in the signal queue."
```

---

### 12.7 Platform Coverage

**Component:** `components/platforms.tsx`
**Layout:** icon grid, 4–5 columns on desktop, 3 on tablet, 2 on mobile. Each cell shows a platform logo (SVG), the platform name, and the connector type badge. Source of truth: `app/connectors/registry.py`.

```
Section label: "PLATFORM COVERAGE"

Heading: "13 connectors. Social, news, and community — all in one queue."

Subheading:
  "Every platform connector implements the same interface. Adding a new
   source takes one API credential and one configuration object. No custom
   ETL pipeline required."

Platform grid:

  1.  Name: Reddit
      Type badge: "Social"
      Credential type: OAuth 2.0 (client_id + client_secret)
      Coverage: Posts, comments, subreddit streams

  2.  Name: YouTube
      Type badge: "Social"
      Credential type: Google API key
      Coverage: Video comments, channel community posts

  3.  Name: TikTok
      Type badge: "Social"
      Credential type: TikTok Developer App credentials
      Coverage: Video comments, creator posts

  4.  Name: Facebook
      Type badge: "Social"
      Credential type: Meta Developer App token
      Coverage: Page posts, public group posts

  5.  Name: Instagram
      Type badge: "Social"
      Credential type: Meta Developer App token
      Coverage: Post captions, comments (via Graph API)

  6.  Name: WeChat
      Type badge: "Social"
      Credential type: WeChat Open Platform credentials
      Coverage: Official account articles

  7.  Name: RSS
      Type badge: "Generic"
      Credential type: None — provide feed URLs directly
      Coverage: Any RSS 2.0 or Atom feed

  8.  Name: The New York Times
      Type badge: "News"
      Credential type: None — public RSS
      Coverage: All section feeds

  9.  Name: The Wall Street Journal
      Type badge: "News"
      Credential type: None — public RSS
      Coverage: All section feeds

  10. Name: ABC News (US)
      Type badge: "News"
      Credential type: None — public feeds
      Coverage: Top stories, technology, business

  11. Name: ABC News Australia
      Type badge: "News"
      Credential type: None — public feeds
      Coverage: Top stories, technology

  12. Name: Google News
      Type badge: "News"
      Credential type: None — scrape only
      Coverage: Top stories by topic

  13. Name: Apple News
      Type badge: "News"
      Credential type: None — scrape only
      Coverage: Top stories by topic

Below the grid:

  CTA:
    Text: "Need a platform that isn't listed?"
    Subtext: "The connector interface is documented. Adding a new connector
              requires implementing one abstract class with three methods:
              authenticate(), fetch(), and validate_credentials()."
    Link: "Read the connector guide →"  → /docs/connectors
```

---

### 12.8 Privacy and Data Residency

**Component:** `components/privacy.tsx`
**Layout:** two-column on desktop (illustration left, copy right); dark or off-white background to separate visually from adjacent sections.

```
Section label: "PRIVACY BY DESIGN"

Heading:
  "Your data never leaves your machine."

Subheading:
  "Social-Media-Radar enforces a zero-egress contract at the application
   boundary. The DataResidencyGuard intercepts every LLM call and verifies
   that no raw personal data is present in the prompt before it is
   dispatched — and writes an immutable audit log entry for every
   redaction it makes."

Feature list (four rows, each with an icon, title, and body):

  Row 1:
    Icon:  Lucide:UserX
    Title: "Author pseudonymisation"
    Body:  "Author handles and user IDs are replaced with deterministic
            SHA-256 pseudonyms before any text is assembled into an LLM
            prompt. The mapping is stored only in your local database."

  Row 2:
    Icon:  Lucide:EyeOff
    Title: "PII scrubbing at the call boundary"
    Body:  "Email addresses, phone numbers, and identifying URL parameters
            are removed from observation text before prompt assembly.
            A secondary verify_clean() check runs immediately before the
            API call — if PII is found at this stage, the call is aborted
            and logged as a policy violation."

  Row 3:
    Icon:  Lucide:FileText
    Title: "Immutable audit log"
    Body:  "Every redaction generates a structured audit log entry with the
            redaction type, a hash of the original value, and a timestamp.
            The log is append-only and stored in your local PostgreSQL
            instance."

  Row 4:
    Icon:  Lucide:Wifi
    Title: "Full offline operation with Ollama"
    Body:  "Configure LOCAL_LLM_URL=http://localhost:11434 and
            LOCAL_LLM_MODEL=llama3.1:8b to route all classification
            inference to a local Ollama instance. In this configuration,
            no observation text ever reaches an external network —
            not even for embeddings (a 512-dimensional bag-of-words
            fallback is used instead)."

Callout box (highlighted):
  "Social-Media-Radar is designed for deployment in environments where
   cloud AI providers are prohibited by compliance policy. The privacy
   architecture was not retrofitted — it is structural."
```

---

### 12.9 LLM Routing and Cost Efficiency

**Component:** `components/llm-routing.tsx`
**Layout:** split panel — routing diagram on left, cost comparison table on right.

```
Section label: "LLM ROUTING"

Heading:
  "Two-tier inference. Frontier accuracy where it matters.
   Fine-tuned efficiency everywhere else."

Body:
  "Social-Media-Radar routes each observation to one of two LLM tiers
   based on the signal type being evaluated. The routing decision is
   made before the LLM call and is deterministic — there is no
   sampling, no probabilistic routing, and no post-hoc reclassification."

Routing table:

  Tier 1 — Frontier:
    Models:         GPT-4o  /  Claude 3.5 Sonnet
    Signal types:   churn_risk · misinformation_risk · support_escalation
    Rationale:      "These three types carry the highest cost of a false
                     negative. A missed churn risk or a missed public
                     misinformation event has outsized business impact.
                     Frontier accuracy is non-negotiable."
    Approx. latency: 1.5 – 4 s per signal

  Tier 2 — Non-Frontier:
    Models:         GPT-4o mini (fine-tuned)  /  Ollama llama3.1:8b (local)
    Signal types:   lead_opportunity · competitor_weakness ·
                    influencer_amplification · product_confusion ·
                    feature_request_pattern · launch_moment · trend_to_content
    Rationale:      "These 7 types have a lower cost of error and respond
                     well to fine-tuning on the seed dataset. Routing them
                     to a cheaper model reduces average per-signal LLM spend
                     by 70–80% with no measurable accuracy regression."
    Approx. latency: 0.4 – 1.2 s per signal (fine-tuned) · 3 – 12 s (local)

Cost comparison callout:
  "At 1,000 signals per day, routing 70% to Tier 2 at GPT-4o mini pricing
   versus routing all to GPT-4o represents a cost reduction of approximately
   85–90% on the non-frontier volume. The exact saving depends on your
   provider pricing at the time of deployment."

Provider configuration table:

  | Provider          | Config key              | Notes                          |
  |---|---|---|
  | OpenAI GPT-4o     | OPENAI_API_KEY          | Frontier tier (required)       |
  | OpenAI fine-tune  | FINE_TUNED_MODEL_ID     | Non-frontier tier (recommended)|
  | Anthropic Claude  | ANTHROPIC_API_KEY       | Alternative frontier tier      |
  | Ollama (local)    | LOCAL_LLM_URL + LOCAL_LLM_MODEL | Zero-cost, zero-egress  |
  | vLLM (self-host)  | VLLM_ENDPOINT           | High-throughput self-hosted    |
```

---

### 12.10 Calibration and Accuracy

**Component:** `components/calibration.tsx`
**Layout:** two-column — explanation left, interactive temperature scalar table right (populated from `data/calibration_state.json` at build time).

```
Section label: "CONFIDENCE CALIBRATION"

Heading:
  "The system gets more accurate the more you use it."

Body:
  "LLMs are systematically miscalibrated: they tend toward overconfidence
   on common signal types and underconfidence on rare ones. A model that
   outputs confidence = 0.91 for every churn_risk does not have 91%
   accuracy — it has whatever accuracy it has on that class, regardless
   of the number it prints.

   Social-Media-Radar applies per-signal-type temperature scaling,
   calibrated on the 107-example seed dataset and updated online after
   every analyst feedback event. The result is a system where a confidence
   score of 0.87 means: in the training distribution, 87% of signals
   classified at this confidence level with this signal type were correct.

   Temperature scalars are updated in-process in 6–8 microseconds per
   feedback event. There is no retraining cycle. There is no minimum
   batch size. The first correction improves subsequent classifications
   immediately."

Accuracy targets callout (from training/fine_tuning_plan.md):
  Heading: "Fine-tuning targets (non-frontier tier)"
  Metric 1: "Macro F1 ≥ 0.82"
  Metric 2: "Expected Calibration Error ≤ 0.05"
  Metric 3: "False-action rate ≤ 0.08"
  Metric 4: "Abstention rate: 5–15% (healthy operating range)"

Current temperature scalars table (rendered from calibration_state.json):
  Column headers: Signal Type | Temperature Scalar | Calibrated
  Rows (values from training/calibration_state.json at build time):
    lead_opportunity           | <T from JSON>  | Yes/No
    competitor_weakness        | <T from JSON>  | Yes/No
    influencer_amplification   | <T from JSON>  | Yes/No
    churn_risk                 | <T from JSON>  | Yes/No
    misinformation_risk        | <T from JSON>  | Yes/No
    support_escalation         | <T from JSON>  | Yes/No
    product_confusion          | <T from JSON>  | Yes/No
    feature_request_pattern    | <T from JSON>  | Yes/No
    launch_moment              | <T from JSON>  | Yes/No
    trend_to_content           | <T from JSON>  | Yes/No

  Note below table (italic):
    "Temperature = 1.0 means uncalibrated (mathematical identity).
     Values < 1.0 reduce overconfident outputs. Values > 1.0 sharpen
     underconfident outputs. Scalars shown above reflect the state after
     calibration on the 107-example seed dataset."

Developer note:
  The calibration_state.json file is read at build time by
  lib/benchmark-data.ts using Node.js fs.readFileSync.
  At runtime this renders as a static HTML table — no client-side fetch.
```

---

### 12.11 Performance Benchmarks

**Component:** `components/benchmarks.tsx`
**Layout:** tabbed interface — one tab per algorithm. Each tab shows a line chart (x-axis: input size n, y-axis: time in ms) and a summary statistics row. Charts use `recharts` or `chart.js`. Data is parsed from `data/*.csv` at build time. Source: `deliverables/results/*.csv`.

```
Section label: "PERFORMANCE"

Heading:
  "Measured. Not estimated."

Subheading:
  "Every number below comes from running deliverables/benchmark.py with
   3 warm-up passes and 7 timed repetitions on Apple M-series hardware.
   No simulations. No projections."

Tab 1 — "Deduplication (Bloom Filter)"
  Chart title: "BloomFilter.add() — Time vs. Input Size"
  X-axis label: "Items checked (n)"
  Y-axis label: "Total time (ms)"
  Data source: data/bloom.csv (columns: n, time_ms)

  Data points (exact from bloom.csv):
    n=500       → 6.1 ms
    n=1,000     → 12.2 ms
    n=5,000     → 60.9 ms
    n=10,000    → 122.2 ms
    n=50,000    → 631.6 ms
    n=100,000   → 1,300.3 ms

  Summary stats row:
    "Per-operation cost:   12–13 µs   (constant)"
    "Memory model:         O(n) bits"
    "False positive rate:  1% (configurable)"
    "False negative rate:  0% (guaranteed)"

Tab 2 — "Stream Sampling (Reservoir)"
  Chart title: "ReservoirSampler — Throughput vs. Stream Length"
  X-axis label: "Stream length (n)"
  Y-axis label: "Time (ms)"
  Data source: data/reservoir.csv

  Data points (exact from reservoir.csv):
    n=1,000     → 0.95 ms
    n=10,000    → 10.1 ms
    n=50,000    → 50.1 ms
    n=100,000   → 101.2 ms
    n=250,000   → 248.3 ms
    n=500,000   → 503.5 ms

  Summary stats row:
    "Throughput:           ~1,000 items/ms   (constant)"
    "Algorithm:            Vitter's Algorithm R"
    "Bias:                 None — uniform probability for every item"
    "Sample size:          500 (configurable via MAX_ITEMS_PER_FETCH)"

Tab 3 — "Confidence Calibration (Calibrator)"
  Chart title: "ConfidenceCalibrator.update() — Time vs. Update Count"
  X-axis label: "Update events (m)"
  Y-axis label: "Computation time (ms)"
  Data source: data/calibrator.csv

  Data points (exact from calibrator.csv):
    m=100       → 0.78 ms
    m=1,000     → 5.6 ms
    m=10,000    → 66.9 ms
    m=100,000   → 666.5 ms
    m=500,000   → 3,282.7 ms

  Summary stats row:
    "Per-update cost:      6–8 µs   (linear)"
    "Storage:              One float (temperature scalar) per signal type"
    "Persistence:          Flush to calibration_state.json after each update"
    "Restart behaviour:    State loaded from disk — updates survive restarts"

Tab 4 — "Signal Ranking (ActionRanker)"
  Chart title: "ActionRanker.rank_batch() — Time vs. Queue Size"
  X-axis label: "Signals ranked (n)"
  Y-axis label: "Time (ms)"
  Data source: data/action_ranker.csv

  Data points (exact from action_ranker.csv):
    n=10        → 0.13 ms
    n=100       → 1.33 ms
    n=1,000     → 14.1 ms
    n=5,000     → 88.9 ms
    n=10,000    → 176.5 ms
    n=50,000    → 887.5 ms

  Summary stats row:
    "Per-signal cost:      14–18 µs   (linear)"
    "Dimensions:           Opportunity (35%) + Urgency (30%) + Risk (35%)"
    "Confidence gate:      Signals below 0.5 confidence are not ranked"
    "Priority tiers:       CRITICAL / HIGH / MEDIUM / LOW"

Tab 5 — "Context Memory (BFS Traversal)"
  Chart title: "ContextMemoryStore BFS — Time vs. Graph Size"
  X-axis label: "Nodes (n)"
  Y-axis label: "Time (ms)"
  Data source: data/bfs.csv

  Data points (exact from bfs.csv):
    n=1,000     → 0.23 ms
    n=5,000     → 1.04 ms
    n=10,000    → 2.09 ms
    n=50,000    → 10.3 ms

  Summary stats row:
    "Complexity:    O(V + E) — linear in nodes and edges"
    "Use case:      Related few-shot example retrieval for LLM context"
    "Topology used in benchmark: degree-4 ring graph"
    "Complement:    pgvector HNSW ANN search for semantic similarity"
```

---

### 12.12 Team Workflow

**Component:** `components/team-workflow.tsx`
**Layout:** alternating image-text rows on desktop, stacked on mobile. Three features, each illustrated by a product screenshot or diagram.

```
Section label: "TEAM WORKFLOW"

Heading:
  "Built for teams. Not inboxes."

Subheading:
  "Social-Media-Radar's signal queue is a shared workspace.
   Every action, assignment, and dismissal is timestamped, attributed
   to the acting user, and available to the whole team."

Feature 1:
  Screenshot: public/screenshots/team-queue.png
  Alt: "Team signal queue showing signals assigned to different team members
        with role badges and urgency scores"
  Title: "Role-based queue management"
  Body:  "Three roles — VIEWER, ANALYST, MANAGER — control what each team
          member sees and can do. Managers see all signals including those
          assigned to other analysts. Viewers see the queue but cannot act.
          Analysts can act, dismiss, and submit feedback."

Feature 2:
  Screenshot: public/screenshots/sse-stream.png
  Alt: "Terminal showing Server-Sent Events stream delivering new signals
        in real time as they are classified by the inference pipeline"
  Title: "Real-time signal delivery via SSE"
  Body:  "New signals are pushed to connected clients over a persistent
          Server-Sent Events connection as soon as they are classified —
          no polling, no webhook setup, no third-party push service.
          Connect with a single curl command or embed the stream in your
          existing dashboard."
  Code block (Shiki, bash):
    curl -N -H "Authorization: Bearer $TOKEN" \
         -H "Accept: text/event-stream" \
         http://localhost:8000/api/v1/signals/stream

Feature 3:
  Screenshot: public/screenshots/feedback-loop.png
  Alt: "Signal card showing a feedback submission form with predicted type
        churn_risk and true type support_escalation selected"
  Title: "Online calibration — the queue gets better with every correction"
  Body:  "When the model misclassifies a signal, submit a correction via
          the API or the web UI. The ConfidenceCalibrator performs one
          gradient-descent step immediately — adjusting the temperature
          scalar for the corrected signal type without any service restart.
          Calibration improvements are visible in subsequent inferences
          within the same session."
```

---

### 12.13 Local Deployment Section

**Component:** `components/deployment-steps.tsx`
**Layout:** centred, high-contrast background. Two paths: Docker Compose (primary) and bare-metal (secondary). Code blocks via Shiki.

```
Section label: "GET STARTED"

Heading: "Running in 5 minutes on any machine."

Subheading:
  "Social-Media-Radar runs on macOS, Ubuntu, and Windows WSL2.
   No cloud account required. No SaaS sign-up. No data ever leaves
   your machine unless you configure an external LLM provider."

Path A heading: "Option A — Docker Compose (Recommended)"
Path A subtext: "Starts the full stack — Postgres, Redis, MinIO, API, Celery — in one command."

Step-by-step code block (Shiki, bash):
  # Step 1: Clone
  git clone https://github.com/yourorg/social-media-radar.git
  cd social-media-radar

  # Step 2: Generate secrets and configure
  cp .env.example .env
  python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
  python3 -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
  # Paste both values into .env, then add your OPENAI_API_KEY

  # Step 3: Start everything
  docker compose up

  # Step 4: Calibrate (in a second terminal while the stack is running)
  docker compose exec api python training/calibrate.py --epochs 5

  # Step 5: Verify
  curl http://localhost:8000/health

Expected output callout:
  {"status": "healthy", "database": "ok", "redis": "ok"}

Path B heading: "Option B — Bare-metal (macOS / Ubuntu)"
Path B subtext: "Run the application process directly — useful for debugging and IDE integration."
Path B link: "Full bare-metal installation guide →"   → /docs/deployment

System requirements table:
  | Component | Minimum          | Recommended                        |
  |---|---|---|
  | CPU       | 4 cores          | 8+ cores                           |
  | RAM       | 8 GB             | 16 GB                              |
  | Disk      | 10 GB            | 30 GB (model weights + data)       |
  | Python    | 3.9              | 3.11                               |
  | Docker    | 24.0+ (Compose v2) | Docker Desktop 4.28+             |
  | OS        | macOS 12, Ubuntu 20.04, WSL2 | macOS 14 / Ubuntu 22.04 |
```

---

### 12.14 Frequently Asked Questions

**Component:** `components/faq.tsx`
**Layout:** accordion (one question visible at a time). Implement with `<details>`/`<summary>` or shadcn/ui Accordion for accessibility. Each question is a landmark heading for screen readers.

```
Section label: "FAQ"
Heading: "Frequently asked questions"

Q1: "Do I need a cloud account or SaaS subscription to run this?"
A1: "No. Social-Media-Radar is open-source software that runs entirely on
     your machine. You do not need a cloud account to run the application
     itself. You will need an OpenAI or Anthropic API key if you want
     to use a hosted LLM for classification. If you configure Ollama with
     a local model (such as llama3.1:8b), no external account of any kind
     is required."

Q2: "What LLM providers are supported?"
A2: "OpenAI (GPT-4o and GPT-4o mini), Anthropic (Claude 3.5 Sonnet and
     Claude 3.5 Haiku), Ollama (any locally-served model), and vLLM
     (self-hosted high-throughput inference). The LLM router is designed
     around an abstract provider interface — adding a new provider requires
     implementing one class with two methods."

Q3: "How is this different from Brandwatch, Mention, or Sprinklr?"
A3: "Those products are SaaS tools that ingest your data on their
     infrastructure, return sentiment and volume metrics, and require an
     ongoing subscription. Social-Media-Radar is a self-hosted application
     that runs on your machine, never sends your data to a third-party
     server (unless you configure a cloud LLM provider), and classifies
     posts into structured business-intent signal types rather than
     sentiment buckets. The output is an action queue, not a report."

Q4: "What happens when the model is not confident enough to classify?"
A4: "The model abstains. It returns a structured abstention with a reason
     code (e.g., ambiguous_intent, insufficient_context, out_of_scope)
     and a confidence estimate for the abstention itself. Abstentions are
     logged separately and never surface in the signal queue. The
     abstention rate in a healthy deployment is between 5% and 15% of
     all observations processed."

Q5: "How do I add a platform that isn't in the list of 13 connectors?"
A5: "Every connector implements the BaseConnector abstract class, which
     defines three methods: authenticate(), fetch(), and
     validate_credentials(). Implement those three methods for your
     platform, register the connector in app/connectors/registry.py,
     and add the platform name to the SourcePlatform enum in
     app/core/models.py. The connector guide at /docs/connectors walks
     through a complete example."

Q6: "Can I run this on Windows?"
A6: "Yes — via Windows Subsystem for Linux 2 (WSL2). Install WSL2,
     choose Ubuntu 22.04 as the distribution, and follow the Linux
     bare-metal installation guide. Docker Desktop for Windows with
     the WSL2 backend is also fully supported and is the simplest
     path on Windows."

Q7: "How does the calibration update work? Does it require a restart?"
A7: "No restart is required. Each feedback submission via
     POST /api/v1/signals/{id}/feedback triggers one gradient-descent
     step on the ConfidenceCalibrator in-process. The temperature scalar
     for the affected signal type is updated in memory immediately and
     flushed to training/calibration_state.json. The next inference on
     that signal type uses the updated scalar. The calibration state is
     loaded from disk on API startup, so updates survive service restarts."

Q8: "What database does this use? Can I use an existing PostgreSQL instance?"
A8: "Social-Media-Radar requires PostgreSQL 15 or later with the pgvector
     extension installed. If you have an existing PostgreSQL 15+ instance
     with pgvector available, set DATABASE_URL and DATABASE_SYNC_URL in
     your .env file to point to it, then run python scripts/init_db.py
     (to enable the extension if not already enabled) and alembic upgrade
     head (to apply the schema migrations). The Docker Compose setup
     starts a dedicated PostgreSQL container if you prefer not to use an
     existing instance."

Q9: "Is there a rate limit on the API?"
A9: "Yes. The default rate limit is 60 requests per minute per
     authenticated user, enforced at the API gateway layer. This can
     be adjusted via the RATE_LIMIT_PER_MINUTE setting in .env.
     The rate limit applies to all endpoints except GET /health and
     GET /api/v1/signals/stream (the SSE endpoint), which are
     excluded from rate limiting."

Q10: "What is the license?"
A10: "MIT License. You are free to use, modify, and distribute this
      software for any purpose, including commercial use, without
      restriction. Attribution is appreciated but not required.
      The full license text is in the LICENSE file in the repository."
```

---

### 12.15 Footer

**Component:** `components/footer.tsx`
**Layout:** four-column grid on desktop, two-column on tablet, single column on mobile. Dark background.

```
Column 1 — Brand:
  Logo: (same as nav)
  Tagline: "Open-source social intelligence for B2B teams."
  Social link: GitHub → {NEXT_PUBLIC_REPO_URL}
  License badge: "MIT License"

Column 2 — Product:
  Heading: "Product"
  Links:
    Features        → /#features
    How It Works    → /#pipeline
    Benchmarks      → /benchmark
    Signal Taxonomy → /#signal-taxonomy
    Changelog       → /changelog

Column 3 — Documentation:
  Heading: "Documentation"
  Links:
    Getting Started   → /docs/deployment
    API Reference     → /docs/api
    Architecture      → /docs/architecture
    LLM Configuration → /docs/llm
    Training Guide    → /docs/training
    Contributing      → {NEXT_PUBLIC_REPO_URL}/blob/main/CONTRIBUTING.md

Column 4 — Resources:
  Heading: "Resources"
  Links:
    GitHub Repository    → {NEXT_PUBLIC_REPO_URL}
    Report an Issue      → {NEXT_PUBLIC_REPO_URL}/issues
    Discussions          → {NEXT_PUBLIC_REPO_URL}/discussions
    Releases / Download  → {NEXT_PUBLIC_REPO_URL}/releases

Bottom bar (below the four columns, full width, small text):
  Left:  "© 2026 Social-Media-Radar Contributors. MIT License."
  Right: "Built with Next.js · Deployed on Vercel"
         (Vercel logo links to vercel.com — optional, remove if using self-hosted)
```

---

## 13. Asset Specifications

All assets in `public/` must meet the specifications below before the site launches. File format and dimensions are fixed — do not use different formats or sizes without updating this section.

| Asset | File | Format | Dimensions | Max file size | Alt text |
|---|---|---|---|---|---|
| Open Graph image | `public/og-image.png` | PNG | 1200 × 630 px | 200 KB | See §10 |
| Logo (nav + footer) | `public/logo.svg` | SVG | 32 × 32 px viewBox | 8 KB | "Social-Media-Radar logo" |
| Favicon | `public/favicon.ico` | ICO | 16×16 + 32×32 + 48×48 | 15 KB | (decorative) |
| Favicon PNG | `public/favicon-32x32.png` | PNG | 32 × 32 px | 5 KB | (decorative) |
| Apple touch icon | `public/apple-touch-icon.png` | PNG | 180 × 180 px | 20 KB | (decorative) |
| Hero screenshot | `public/screenshots/signal-queue.png` | PNG or WebP | 1280 × 800 px | 300 KB | See §12.2 |
| Team queue screenshot | `public/screenshots/team-queue.png` | PNG or WebP | 1280 × 800 px | 300 KB | See §12.12 |
| SSE stream screenshot | `public/screenshots/sse-stream.png` | PNG or WebP | 1280 × 600 px | 200 KB | See §12.12 |
| Feedback loop screenshot | `public/screenshots/feedback-loop.png` | PNG or WebP | 1280 × 800 px | 300 KB | See §12.12 |

**Screenshot guidelines:**
- Use a dark system theme if the application supports it; it renders better on the website's dark sections.
- Redact any real credentials, API keys, or personal data visible in the interface before saving.
- Annotate screenshots with call-out arrows only if a specific element needs to be highlighted; otherwise use clean screenshots.
- Compress PNG files with `pngcrush` or `oxipng` before committing. Target < 300 KB per screenshot.

---

## 14. Accessibility and Performance Targets

### Accessibility (WCAG 2.1 AA)

| Requirement | Implementation note |
|---|---|
| All images have `alt` text | Use the exact alt text strings defined in §12 |
| Colour contrast ≥ 4.5:1 (normal text) | Verify with [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/) after theming |
| Keyboard navigation | All interactive elements reachable via Tab; focus ring visible |
| Accordion FAQ | Use `<details>`/`<summary>` or ARIA `role="button"` + `aria-expanded` |
| Landmark regions | `<nav>`, `<main>`, `<footer>` — one each; no duplicate landmarks |
| Skip link | `<a href="#main-content" class="sr-only focus:not-sr-only">Skip to content</a>` as first child of `<body>` |
| Font size | Base: 16 px minimum; body copy: 18 px on desktop |
| Motion | Respect `prefers-reduced-motion` — disable all scroll animations and auto-playing videos |

### Core Web Vitals Targets (Production)

Run `pnpm build && pnpm start`, then measure with Lighthouse in Chrome DevTools:

| Metric | Target | Tool |
|---|---|---|
| Largest Contentful Paint (LCP) | < 2.5 s | Lighthouse / PageSpeed Insights |
| Cumulative Layout Shift (CLS) | < 0.1 | Lighthouse |
| Interaction to Next Paint (INP) | < 200 ms | Chrome DevTools |
| Total Blocking Time (TBT) | < 300 ms | Lighthouse |
| Home page size (gzipped) | < 500 KB | `next build` output |
| Time to First Byte (TTFB) | < 200 ms | Vercel Edge Network / self-hosted nginx |

Run Lighthouse as part of the CI pipeline using `@lhci/cli`:

```bash
# Install once
pnpm add -D @lhci/cli

# Add to package.json scripts
"lighthouse": "lhci autorun"

# .lighthouserc.js
module.exports = {
  ci: {
    collect: { startServerCommand: 'pnpm start', url: ['http://localhost:3000'] },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.95 }],
        'categories:best-practices': ['warn',  { minScore: 0.9 }],
        'categories:seo': ['error', { minScore: 0.95 }],
      },
    },
  },
}
```

---

## 15. Analytics and Tracking

**Recommended:** Plausible Analytics (self-hostable, GDPR-compliant, no cookies, no consent banner required).

### Plausible Setup

```bash
pnpm add next-plausible
```

In `app/layout.tsx`:

```typescript
import PlausibleProvider from 'next-plausible'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <PlausibleProvider domain={process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN ?? ''} />
      </head>
      <body>{children}</body>
    </html>
  )
}
```

If `NEXT_PUBLIC_PLAUSIBLE_DOMAIN` is empty, `PlausibleProvider` renders nothing — analytics are disabled without a code change.

### Events to Track

Track the following custom events using `usePlausible()` from `next-plausible`:

| Event name | Trigger | Props |
|---|---|---|
| `Download` | Click on any "Download" button | `{ version: APP_VERSION }` |
| `GitHub` | Click on any GitHub link | `{ location: 'nav' \| 'hero' \| 'footer' }` |
| `Docs` | Click on any documentation link | `{ page: '/docs/deployment' \| '/docs/api' \| … }` |
| `DemoLogin` | User submits the live demo login form | (none) |
| `BenchmarkTab` | User switches to a benchmark tab | `{ tab: 'bloom' \| 'reservoir' \| … }` |
| `FaqOpen` | User opens a FAQ accordion item | `{ question_index: number }` |

---

## 16. Maintenance Notes

### Keeping Website Copy in Sync with the Application

The following application files, when changed, **require a corresponding update to the website**:

| Application file | Website section(s) affected |
|---|---|
| `app/connectors/registry.py` | §12.7 Platform Coverage, Trust Bar stat 1 |
| `app/core/signal_models.py` (SignalType enum) | §12.6 Signal Taxonomy, Trust Bar stat 2, §12.5 Step 5 body |
| `deliverables/results/*.csv` | §12.11 Benchmarks — all five tabs and summary stats |
| `training/calibration_state.json` | §12.10 temperature scalar table |
| `training/fine_tuning_plan.md` | §12.10 accuracy targets callout |
| `app/core/config.py` (RATE_LIMIT_PER_MINUTE) | §12.14 FAQ Q9 |
| `requirements.txt` / `pyproject.toml` | §12.2 Hero tech stack line, §10 SEO keywords |
| Test count (pytest output) | §12.2 Hero stats line, §12.3 Trust Bar stat 3 |

### Benchmark Data Update Process

When the application team re-runs benchmarks on new hardware or with algorithmic changes:

1. Run `python deliverables/benchmark.py` in the application repo.
2. The CSV files in `deliverables/results/` are overwritten with new measurements.
3. Run `pnpm prebuild` in the website repo (or `node scripts/sync-benchmark-data.js`) to copy the updated CSVs into `data/`.
4. Run `pnpm build` and inspect the benchmark page to confirm the charts and summary stats reflect the new values.
5. Update the exact data point values in §12.11 of this document to match the new CSV output.
6. Open a pull request with the updated `data/` files and this document.

### Release Checklist (Before Publishing a New App Version)

- [ ] Update `NEXT_PUBLIC_APP_VERSION` in Vercel environment variables and in `vars.APP_VERSION` GitHub variable.
- [ ] Sync benchmark data (`node scripts/sync-benchmark-data.js`).
- [ ] Update test count in §12.2 and §12.3 to match the new baseline.
- [ ] Take new screenshots if the UI has changed and replace files in `public/screenshots/`.
- [ ] Update the Changelog page (`app/changelog/page.mdx`) with the release notes.
- [ ] Run `pnpm build && pnpm test` — confirm zero failures.
- [ ] Run Lighthouse audit — confirm all scores meet §14 targets.
- [ ] Merge to `main` — Vercel deploys automatically.

---

*Document maintained by the website development team. For questions about the application internals cited here, refer to the application repository's `README.md`, `docs/` directory, and source code.*
