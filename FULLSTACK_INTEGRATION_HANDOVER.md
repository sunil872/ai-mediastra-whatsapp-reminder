# MediAstra RefillCare — Integration Handover for NestJS & Angular 19 Team

> **Audience:** NestJS Backend Engineers, Angular 19 (Fuse Admin) Frontend Engineers, and DevOps / Cloud Engineers.  
> **Environment:** TypeScript across stack (**NestJS 10**, Prisma 5.22, PostgreSQL 15) & **Angular 19** (Fuse Admin, Tailwind CSS 3, Material) deployed on **AWS (ap-south-1 Mumbai)** via **Amazon ECS Fargate**, **Amazon ECR**, **S3 + CloudFront**, and **Docker**.  
> **Tooling Note for AI (Cursor / Antigravity / Claude Code):** This document contains the full architecture, internal business logic, database isolation strategy, exact REST API contracts, and ready-to-use NestJS & Angular 19 code for the **MediAstra RefillCare AI Microservice**.

---

## 1. Executive Summary & Infrastructure Alignment

* **Product:** MediAstra RefillCare (*"Where Pharmacy Meets Intelligence"*)
* **Purpose:** Clinical predictive engine that calculates exact prescription refill dates for chronic patients and automates multi-stage WhatsApp reminders.
* **Status:** 100% built, verified, and operational on real pharmacy history (827,000+ sales transactions).
* **Infrastructure Alignment with MediAstra Platform:**
  * **Compute (AWS ECS Fargate):** RefillCare packages into a standard Docker container pushed to **Amazon ECR** and executed as an ECS task in `ap-south-1` alongside the NestJS API container.
  * **Database (PostgreSQL 15):** Connects to the same PostgreSQL 15 database instance using **schema isolation** (`refillcare_control` and `s_<store_key>`). It does **not** alter or touch any Prisma tables in the `public` schema.
  * **File Storage (Hybrid / Local Disk Mount):** Matches your NestJS storage pattern (`./uploads:/app/uploads` volume mount). Uploaded sales sheets and trained models are stored in `/data/refillcare` on the container's volume (or S3 if `USE_S3_STORAGE=true`).
  * **WhatsApp Integration:** RefillCare is built natively around the **Xinno WhatsApp Gateway (`whatsapp.xinno.in`)**, which MediAstra already uses for outbound communications.
  * **Frontend Delivery:** Served natively through MediAstra's Angular 19 build deployed via **Amazon S3 + CloudFront** (`qa.mediastracloud.com` / `app.mediastracloud.com`).

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            MEDIASTRA APPLICATION STACK (AWS)                             │
│                                                                                          │
│  [Angular 19 Frontend] (Fuse Admin, Tailwind 3, S3 + CloudFront: qa.mediastracloud.com)  │
│          │                                                                               │
│          ▼ HTTPS / REST (JWT & X-Store-Id)                                               │
│  [NestJS 10 API] (ECS Fargate: api.mediastracloud.com / Express, Prisma 5.22, CQRS)     │
│          │                                                                               │
│          ├───────────────────────────────────────────────────────┐                       │
│          ▼ Internal HTTP (ECS Service Discovery / Port 8000)      ▼ Prisma 5.22           │
│  [RefillCare Python Container] (FastAPI, Scikit-learn, Port 8000) [PostgreSQL 15 (RDS)]  │
│          │                                                        ├── public (Prisma)    │
│          ├─────────────── Persistent Volume (/data/refillcare)    ├── refillcare_control │
│          │                                                        └── s_<store_key>      │
│          ▼ Outbound HTTPS                                                                │
│  [Xinno WhatsApp Gateway] (whatsapp.xinno.in / Meta Cloud API)                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive: Current Refill Reminder Implementation

This section details the clinical logic, prediction routing, multi-stage lifecycle, and safety mechanisms **already implemented and working** in RefillCare.

```
                                  CUSTOMER BILLING TRANSACTION
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     CLINICAL DRUG CLASSIFIER     │
                              └────────────────┬─────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
                [ACUTE MEDICINE]                              [CHRONIC MEDICINE]
            (Cough Syrups, Antibiotics)                   (Diabetes, BP, Thyroid, Heart)
                       │                                               │
                [NO REMINDER]                                          ▼
                                                       ┌───────────────────────────────┐
                                                       │    DECISION ROUTING ENGINE    │
                                                       └───────────────┬───────────────┘
                                                                       │
                                   ┌───────────────────────────────────┴───────────────────────────────────┐
                                   ▼                                                                       ▼
                         [PATH A: ESTABLISHED]                                                   [PATH B: NEW PATIENT]
                       (≥ 3 Historical Purchases)                                                (1 - 2 Purchases Only)
                                   │                                                                       │
                                   ▼                                                                       ▼
                        Machine Learning Predictor                                              Days of Supply (DOS)
                    (Cadence, Habit, Drift, Seasonality)                                     (Pack Units ÷ Daily Dosage)
                                   │                                                                       │
                                   └───────────────────────────────┬───────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                       [PREDICTED REFILL DATE]
                                                                   │
                                                                   ▼
                                                    ┌─────────────────────────────┐
                                                    │    7-STAGE OUTREACH CYCLE   │
                                                    │  -7d, -3d, -1d, 0d, +2d...  │
                                                    └──────────────┬──────────────┘
                                                                   │
                                       ┌───────────────────────────┴───────────────────────────┐
                                       ▼                                                       ▼
                             [MED-SYNC ENGINE]                                      [REPURCHASE DETECTOR]
                        Bundles 2-5 medicines due                                 Customer buys medicine early?
                        into 1 WhatsApp message                                   Cancels all pending stages!
```

---

### A. The 3-Path Clinical Decision Routing (`unified_engine.py`)

Each billing row generates a composite patient-medicine key: `customer_item_key` (`customerId` + `itemId`). The engine routes the record into one of three paths:

1. **Path A (Established Chronic Patient):**
   * **Trigger:** Patient has $\ge 3$ verified historical purchases of that chronic medicine.
   * **Algorithm:** Analyzes interval cadences, purchase stability tiers (`HIGH`, `MEDIUM_SAFE`, `MEDIUM_RISK`), and interval drift. Uses the pre-trained HistGradientBoosting predictor (44 clinical features) to calculate the exact calendar date when the patient will exhaust their current supply.
2. **Path B (New or Short-History Patient):**
   * **Trigger:** Patient has 1 or 2 purchases (insufficient data for statistical cadence modeling).
   * **Algorithm:** Computes **Estimated Days of Supply (DOS)**:
     $$\text{Days of Supply} = \frac{\text{Pack Size} \times (\text{Quantity} + \text{Free Quantity})}{\text{Estimated Daily Dosage}}$$
     *Example:* If a patient buys 1 strip of 30 tablets with 1 tablet/day dosage, DOS = 30 days. Predicted refill date = `invoice_date + 30 days`.
3. **Ineligible / Exception Queue:**
   * **Trigger:** One-off acute drugs, extreme quantity anomalies (e.g., buying 50 strips at once), pediatric dosage flags, or churned patients ($> 180$ days since expected refill).
   * **Action:** Excluded from auto-dispatch or routed to the **Pharmacist Exception Queue** (`/api/v1/review/queue`) for manual review.

---

### B. The 7-Stage Structured Reminder Lifecycle (`reminder_lifecycle.py`)

Outreach is structured across 7 milestone stages relative to the predicted refill date:

| Stage Name | Offset | Intent Category | Customer Message Text | Status Progression |
|---|---|---|---|---|
| **Stage -7** | -7 Days | `DUE_REFILL` | Early reminder: "Your medicine refill is due in 7 days." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage -3** | -3 Days | `DUE_REFILL` | Advance reminder: "Refill due in 3 days. Order now." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage -1** | -1 Day | `DUE_REFILL` | Urgent notice: "1 day of medicine remaining." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage 0** | Due Day | `DUE_REFILL` | Due today: "Your regular medicine supply runs out today." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage +2** | +2 Days | `REFILL_FOLLOW_UP` | Follow-up: "Did you miss your refill? Avoid missing doses." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage +5** | +5 Days | `REFILL_FOLLOW_UP` | Clinical check-in: "It has been 5 days since your refill date." | `PENDING` ➔ `APPROVED` ➔ `SENT` |
| **Stage +45**| +45 Days| `LAPSED_REENGAGEMENT` | Win-back: "Checking in on your chronic health routine." | `PENDING` ➔ `APPROVED` ➔ `SENT` |

---

### C. The 3 Safety & Anti-Spam Mechanisms

1. **Repurchase Auto-Cancellation:**
   * If a patient visits the store on Day -2 and purchases their medicine early, the system detects the new bill upon ingestion.
   * It **instantly marks the active cycle as `COMPLETED` and cancels all future pending stages** (`Stage -1`, `Stage 0`, `Stage +2`, etc.).
   * The patient never receives an outdated reminder after they have already purchased.
2. **Med-Sync (Prescription Bundling):**
   * Chronic patients frequently take 2 to 5 concurrent medications (e.g. Metformin + Telmisartan + Atorvastatin).
   * If these items have refill dates within $\pm 4$ days of each other, the Med-Sync engine (`med_sync.py`) groups them into **one consolidated message**.
   * Rather than sending 3 separate notifications during the week, the patient receives **one single professional message** listing all due medicines.
3. **Tenant-Wide WhatsApp Opt-Out (STOP / START):**
   * If a customer replies "STOP", "UNSUBSCRIBE", or "CANCEL", the webhook automatically flags the phone number in `whatsapp_consent`.
   * The dispatcher checks this table before every send; all future automated messages are permanently suppressed across all stores of that tenant.

---

### D. Standardized WhatsApp Message Template (`template_formatter.py`)

Outbound messages dispatched via **Xinno (`whatsapp.xinno.in`)** follow this approved template structure:

```
Dear *{{1}}* Garu,

This is a friendly reminder from *{{2}}* regarding your regular medicine refill.

{{3}}

💊 Medicines:
{{4}}

If you need a refill, please contact us or place your order with our pharmacy.

📞 Contact: {{5}}

Thank you for choosing *{{6}}*.
🙏 We are happy to serve you.
```

* **`{{1}}` Patient Name:** e.g., "Rajendra Reddy"
* **`{{2}}` Pharmacy Name:** e.g., "NEHA PHARMACY - DURGA PHC"
* **`{{3}}` Stage Category Text:**
  * For Due stages: *"Your regular medicine refill is due soon."*
  * For Follow-up stages: *"We’re following up regarding your regular medicine refill."*
  * For Lapsed stages: *"We’re checking in regarding your regular medicine refill."*
* **`{{4}}` Formatted Medicine List (Med-Sync):**
  * `• TELMISARTAN 40MG (1 Strip)`
  * `• METFORMIN 500MG (2 Strips)`
* **`{{5}}` Store Phone Number:** For direct ordering via phone or WhatsApp.
* **`{{6}}` Pharmacy Sign-off Name.**

---

### E. Data Ingestion & Sanitization (The 11 Ingested Fields)

When sales data is uploaded, RefillCare strips out all pricing, tax, margins, and cost data. It retains strictly these 11 clinical fields:

| Field Name | Description & Purpose |
|---|---|
| `invoice_date` | Purchase date; used to determine consumption interval timelines. |
| `quantity` | Units purchased; multiplied by pack size to determine days of supply. |
| `MOBILE_NO` | WhatsApp recipient number and patient deduplication key. |
| `customerName` | Patient name for message salutations. |
| `itemName` | Medication name; used for chronic classification and message text. |
| `invoice_number` | Distinguishes retail customer sales (`S0/`) from wholesale B2B (`SB/`). |
| `itemCode` | Product code; used for master catalogue mapping. |
| `packing` | Number of tablets/capsules per strip (e.g. `1X10`, `1X15`). |
| `freeQuantity` | Free bonus strips that extend patient supply. |
| `customerId` | Unique patient identifier. |
| `itemId` | Unique product identifier. |

---

## 3. Database Isolation Strategy (PostgreSQL 15 & Prisma 5.22)

Your NestJS backend uses Prisma 5.22 against PostgreSQL 15. RefillCare will **not** conflict with your Prisma schema:

| Schema Name | Owner | Managed By | Purpose |
|---|---|---|---|
| `public` | NestJS Backend | Prisma (`prisma migrate`) | Core MediAstra tables: users, auth, billing, store masters. |
| `refillcare_control` | RefillCare Microservice | Python Migrations (`database.migrations`) | Global tenant-to-store maps, WhatsApp consent/opt-outs, background jobs. |
| `s_<store_key>` | RefillCare Microservice | Python Migrations (`database.migrations`) | Isolated store tables: `sales_transactions`, `refill_decisions`, `reminder_cycles`, `reminder_stages`, `import_batches`. |

---

## 4. REST API Contract (Endpoints Ready to Consume)

* **Base URL (Docker internal):** `http://refillcare-api:8000/api/v1`
* **Base URL (Gateway/Proxy):** `https://qa.mediastracloud.com/api/v1/refill`
* **Interactive OpenAPI/Swagger Docs:** `http://localhost:8000/docs`
* **Required Headers:**
  * `X-Store-Id: NEHA STORE` (or store key)
  * `X-API-Key: mda_qa_sec_8923fbc0189a7` (Configured in `.env`)
  * `Content-Type: application/json`

### Key Endpoints

#### 1. `GET /api/v1/reminders/daily` — Get Today's Due Reminders
```json
[
  {
    "id": "rem_stage_9182",
    "customer_name": "Rajendra Reddy",
    "mobile_no": "9848012345",
    "item_name": "TELMISARTAN 40MG",
    "predicted_refill_date": "2026-11-14",
    "days_remaining": 2,
    "stage": "-2_DAYS",
    "status": "PENDING"
  }
]
```

#### 2. `POST /api/reminders/{id}/approve` & `POST /api/reminders/{id}/reject`
Approves or rejects a scheduled reminder message.
```json
// Response (200 OK)
{ "status": "success", "reminder_id": "rem_stage_9182", "action": "APPROVED" }
```

#### 3. `POST /api/v1/whatsapp/send-single` — Dispatch Message
```json
// Request Body:
{ "reminder_id": "rem_stage_9182" }

// Response (200 OK):
{ "status": "dispatched", "message_id": "wamid.HBgMOTE...", "phone": "9848012345" }
```

#### 4. `POST /api/v1/sales/uploads` — Upload Sales Data (CSV / Excel)
* **Header:** `Content-Type: multipart/form-data`
* **Body:** `file: <binary .csv or .xlsx>`
* **Response (202 Accepted):**
```json
{
  "batch_id": "batch_20261112_01",
  "status": "QUEUED",
  "message": "File received. Processing in background."
}
```

#### 5. `GET /livez` & `GET /readyz` — Health Probes
Returns `200 OK {"status": "alive"}` for AWS ALB / monitoring.

---

## 5. NestJS Integration Guide (Backend)

In your NestJS application (`backend/`), create a lightweight client module to proxy requests or integrate RefillCare into your CQRS handlers.

### 1. `refillcare.service.ts`
```typescript
import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class RefillCareService {
  private readonly logger = new Logger(RefillCareService.name);
  private readonly baseUrl = process.env.REFILLCARE_API_URL || 'http://refillcare-api:8000/api/v1';
  private readonly apiKey = process.env.REFILLCARE_API_KEY || 'mda_qa_sec_8923fbc0189a7';

  constructor(private readonly httpService: HttpService) {}

  private getHeaders(storeId: string) {
    return {
      'X-Store-Id': storeId,
      'X-API-Key': this.apiKey,
      'Content-Type': 'application/json',
    };
  }

  async getDailyReminders(storeId: string) {
    const { data } = await firstValueFrom(
      this.httpService.get(`${this.baseUrl}/reminders/daily`, {
        headers: this.getHeaders(storeId),
      }),
    );
    return data;
  }

  async approveReminder(reminderId: string, storeId: string) {
    const { data } = await firstValueFrom(
      this.httpService.post(
        `${this.baseUrl}/reminders/${reminderId}/approve`,
        {},
        { headers: this.getHeaders(storeId) },
      ),
    );
    return data;
  }

  async dispatchWhatsApp(reminderId: string, storeId: string) {
    const { data } = await firstValueFrom(
      this.httpService.post(
        `${this.baseUrl}/whatsapp/send-single`,
        { reminder_id: reminderId },
        { headers: this.getHeaders(storeId) },
      ),
    );
    return data;
  }
}
```

### 2. `refillcare.controller.ts`
```typescript
import { Controller, Get, Post, Param, Headers, UseGuards } from '@nestjs/common';
import { RefillCareService } from './refillcare.service';

@Controller('refill')
export class RefillCareController {
  constructor(private readonly refillService: RefillCareService) {}

  @Get('daily-reminders')
  async getDailyReminders(@Headers('x-store-id') storeId: string) {
    return this.refillService.getDailyReminders(storeId || 'NEHA STORE');
  }

  @Post('reminders/:id/approve')
  async approveReminder(@Param('id') id: string, @Headers('x-store-id') storeId: string) {
    return this.refillService.approveReminder(id, storeId || 'NEHA STORE');
  }

  @Post('whatsapp/send/:id')
  async sendWhatsApp(@Param('id') id: string, @Headers('x-store-id') storeId: string) {
    return this.refillService.dispatchWhatsApp(id, storeId || 'NEHA STORE');
  }
}
```

---

## 6. Angular 19 (Fuse Admin) Frontend Integration Guide

MediAstra uses **Angular 19, Fuse Admin Template, Angular Material, and Tailwind CSS 3**.

### 1. Register Navigation in Fuse Navigation Config
Open your navigation item file (e.g. `src/app/core/navigation/navigation.data.ts` or where the **Utilities ➔ Mini Apps** menu is defined):

```typescript
// Add under Utilities -> Mini Apps (alongside Forecast, Payment Alerts, etc.)
{
  id: 'refill-reminders',
  title: 'Refill Reminders',
  type: 'basic',
  icon: 'heroicons_outline:bell-alert',
  link: '/utilities/mini-apps/refill-reminders'
}
```

### 2. Angular 19 Component (`refill-reminder.component.ts`)
Using Angular 19 standalone component with Angular Material and Tailwind CSS 3:

```typescript
import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatCardModule } from '@angular/material/card';

export interface ReminderItem {
  id: string;
  customer_name: string;
  mobile_no: string;
  item_name: string;
  predicted_refill_date: string;
  days_remaining: number;
  stage: string;
  status: 'PENDING' | 'APPROVED' | 'DISPATCHED' | 'REJECTED';
}

@Component({
  selector: 'app-refill-reminder',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatTableModule,
    MatIconModule,
    MatChipsModule,
    MatCardModule
  ],
  templateUrl: './refill-reminder.component.html'
})
export class RefillReminderComponent implements OnInit {
  private http = inject(HttpClient);
  
  displayedColumns: string[] = ['customer_name', 'mobile_no', 'item_name', 'predicted_refill_date', 'status', 'actions'];
  reminders = signal<ReminderItem[]>([]);
  searchQuery = signal<string>('');
  currentStore = 'NEHA STORE';

  ngOnInit(): void {
    this.loadReminders();
  }

  loadReminders(): void {
    const headers = new HttpHeaders({ 'X-Store-Id': this.currentStore });
    this.http.get<ReminderItem[]>('/api/v1/reminders/daily', { headers }).subscribe({
      next: (data) => this.reminders.set(data || []),
      error: (err) => console.error('Failed to load reminders', err)
    });
  }

  approve(item: ReminderItem): void {
    const headers = new HttpHeaders({ 'X-Store-Id': this.currentStore });
    this.http.post(`/api/v1/reminders/${item.id}/approve`, {}, { headers }).subscribe(() => {
      item.status = 'APPROVED';
    });
  }

  sendWhatsApp(item: ReminderItem): void {
    const headers = new HttpHeaders({ 'X-Store-Id': this.currentStore });
    this.http.post('/api/v1/whatsapp/send-single', { reminder_id: item.id }, { headers }).subscribe(() => {
      item.status = 'DISPATCHED';
    });
  }
}
```

### 3. Angular 19 Template (`refill-reminder.component.html`)
Styled with **Tailwind CSS 3** and **Fuse Card** styling:

```html
<div class="flex flex-col flex-auto min-w-0 p-6 bg-slate-50">
  
  <!-- Page Header -->
  <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 gap-4">
    <div>
      <div class="text-3xl font-bold tracking-tight text-slate-900">Refill Reminders</div>
      <div class="text-sm text-slate-500 mt-1">AI-driven predictive refill queue for chronic medication patients.</div>
    </div>
    <div class="flex items-center gap-3">
      <button mat-stroked-button (click)="loadReminders()" class="bg-white">
        <mat-icon class="mr-1">refresh</mat-icon> Refresh
      </button>
    </div>
  </div>

  <!-- KPI Summary Cards -->
  <div class="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-6">
    <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm flex items-center justify-between">
      <div>
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500">Total Due Today</div>
        <div class="text-3xl font-extrabold text-slate-900 mt-1">{{ reminders().length }}</div>
      </div>
      <div class="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center text-xl font-bold">💊</div>
    </div>

    <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm flex items-center justify-between">
      <div>
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500">Active Store</div>
        <div class="text-xl font-bold text-slate-800 mt-1">{{ currentStore }}</div>
      </div>
      <div class="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center text-xl font-bold">🏪</div>
    </div>

    <div class="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm flex items-center justify-between">
      <div>
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500">WhatsApp Engine</div>
        <div class="text-sm font-semibold text-emerald-600 mt-1">Xinno Gateway Active</div>
      </div>
      <div class="w-12 h-12 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center text-xl font-bold">📱</div>
    </div>
  </div>

  <!-- Reminders Table Card -->
  <div class="bg-white rounded-2xl border border-slate-200/80 shadow-sm overflow-hidden">
    <table mat-table [dataSource]="reminders()" class="w-full">
      
      <ng-container matColumnDef="customer_name">
        <th mat-header-cell *matHeaderCellDef class="font-semibold text-slate-700">Patient</th>
        <td mat-cell *matCellDef="let item" class="font-medium text-slate-900">{{ item.customer_name }}</td>
      </ng-container>

      <ng-container matColumnDef="mobile_no">
        <th mat-header-cell *matHeaderCellDef class="font-semibold text-slate-700">Mobile</th>
        <td mat-cell *matCellDef="let item" class="font-mono text-xs text-slate-600">{{ item.mobile_no }}</td>
      </ng-container>

      <ng-container matColumnDef="item_name">
        <th mat-header-cell *matHeaderCellDef class="font-semibold text-slate-700">Medication</th>
        <td mat-cell *matCellDef="let item" class="text-slate-800">{{ item.item_name }}</td>
      </ng-container>

      <ng-container matColumnDef="predicted_refill_date">
        <th mat-header-cell *matHeaderCellDef class="font-semibold text-slate-700">Refill Due</th>
        <td mat-cell *matCellDef="let item">
          <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700">
            {{ item.predicted_refill_date }} ({{ item.days_remaining }}d)
          </span>
        </td>
      </ng-container>

      <ng-container matColumnDef="status">
        <th mat-header-cell *matHeaderCellDef class="font-semibold text-slate-700">Status</th>
        <td mat-cell *matCellDef="let item">
          <span [ngClass]="{
            'bg-amber-100 text-amber-800': item.status === 'PENDING',
            'bg-blue-100 text-blue-800': item.status === 'APPROVED',
            'bg-emerald-100 text-emerald-800': item.status === 'DISPATCHED'
          }" class="px-2.5 py-0.5 rounded-full text-xs font-semibold">
            {{ item.status }}
          </span>
        </td>
      </ng-container>

      <ng-container matColumnDef="actions">
        <th mat-header-cell *matHeaderCellDef class="text-right font-semibold text-slate-700">Actions</th>
        <td mat-cell *matCellDef="let item" class="text-right space-x-2">
          <button *ngIf="item.status === 'PENDING'" (click)="approve(item)" class="px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium rounded-lg">
            Approve
          </button>
          <button (click)="sendWhatsApp(item)" class="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-medium rounded-lg shadow-sm">
            Send WhatsApp
          </button>
        </td>
      </ng-container>

      <tr mat-header-row *matHeaderRowDef="displayedColumns" class="bg-slate-50/80"></tr>
      <tr mat-row *matRowDef="let row; columns: displayedColumns;" class="hover:bg-slate-50/60 transition-colors"></tr>
    </table>
  </div>

</div>
```

---

## 7. AWS ECS Fargate & CI/CD Deployment Guide

The deployment matches your existing GitHub Actions and Amazon ECS Fargate workflow in region `ap-south-1` (Mumbai):

### 1. Build and Push Image to Amazon ECR
```bash
# Authenticate to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com

# Build RefillCare image
docker build -t refillcare:latest .

# Tag and push
docker tag refillcare:latest <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/mediastra-refillcare:latest
docker push <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/mediastra-refillcare:latest
```

### 2. Database Migration Task (One-Off ECS Task)
Just like your Prisma migration container runs as a one-off task before deploy, run the RefillCare migration task:
```bash
python -m database.migrations --all-stores
```

### 3. ECS Task Definition (Fargate Container Spec)
Add RefillCare to your ECS Task or launch as an ECS Service:
* **Container Name:** `refillcare-api`
* **Image:** `<aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/mediastra-refillcare:latest`
* **Port:** `8000`
* **Environment Variables:**
  * `DATABASE_URL`: `postgresql+psycopg://user:pass@rds-cluster.mediastracloud.com:5432/mediastra_db`
  * `REFILLCARE_ENV`: `production`
  * `REFILLCARE_API_KEY`: `mda_qa_sec_8923fbc0189a7`
  * `XINNO_API_KEY`: `<your_xinno_key>`
  * `XINNO_WABA_NUMBER`: `<your_waba_number>`
* **Volume Mount:** Mount an EFS or persistent storage volume to `/data/refillcare` (matching your `./uploads:/app/uploads` volume strategy).

---

## 8. Verification & Smoke Test

1. **Verify Health Endpoint:**
   ```bash
   curl http://localhost:8000/livez
   # Output: {"status":"alive"}
   ```
2. **Verify Database Connectivity:**
   ```bash
   curl http://localhost:8000/readyz
   # Output: {"status":"ready","database":"connected"}
   ```
3. **Open MediAstra Frontend (`localhost:4200`):**
   Navigate to **Utilities ➔ Mini Apps ➔ Refill Reminders**. Confirm that the Angular Material table displays due patients with their calculated refill dates and days-of-supply countdowns.
