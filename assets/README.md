# Brand & Media Assets

This directory contains the official brand assets, logos, and vector media for Mediastra RefillCare.

## Directory Structure

```
assets/
├── images/
│   ├── MediAstra_Logo.png   # Primary horizontal brand logo (RGBA transparent, 536x344)
│   ├── MediAstra_Icon.png   # Centered square brand icon (RGBA transparent, 512x512)
│   └── favicon.ico          # Multi-resolution ICO (16x16, 32x32, 48x48, 64x64)
├── whatsapp.svg             # WhatsApp official brand icon
└── README.md                # Documentation of asset usage
```

## Usage Across the Platform

1. **Streamlit Frontend (`app_refillcare.py`)**:
   - `page_icon`: Loaded from `assets/images/MediAstra_Icon.png` (renders as browser tab icon).
   - `st.logo`: Sidebar logo loaded from `assets/images/MediAstra_Logo.png`.
   - Header Banner: Rendered inline with glowing drop shadow via base64 data URI.

2. **Web UI (`frontend/`)**:
   - `favicon.ico` & `favicon.png`: Loaded by `<link rel="icon">` and `<link rel="shortcut icon">`.
   - Sidebar Brand: Displayed in `.sidebar-brand .brand-icon-box` via `/static/assets/images/MediAstra_Logo.png`.

3. **FastAPI Backend (`api/main.py`)**:
   - Static Mount: `/assets` mounts `assets/` directory.
   - Favicon Route: `GET /favicon.ico` serves `favicon.ico` for browser requests (including Swagger `/docs` and Redoc).

