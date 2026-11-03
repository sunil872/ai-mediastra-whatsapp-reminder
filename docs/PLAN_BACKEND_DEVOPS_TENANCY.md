# Enterprise Multi-Tenant Architecture & DevOps Deployment Plan: Mediastra RefillCare

> **Target Audience:** Mediastra Engineering Management, Backend Team (TypeScript/Node.js/Prism), and DevOps/Cloud Infrastructure Team.  
> **System Role:** Multi-Tenant Medication Adherence, Refill Forecasting, and Clinical WhatsApp Dispatch Microservice.  
> **Status:** Production-Ready Architecture & Deployment Specification.

---

## 1. Executive Summary & Architectural Vision

The Mediastra platform currently operates a live enterprise stack featuring an **Angular / TypeScript Single Page Application (SPA)**, a **Node.js / Prism Backend**, and **AWS RDS PostgreSQL** for relational persistence.

The **RefillCare AI Engine** introduces machine-learning-driven chronic medication refill forecasting, 6-stage adherence lifecycle scheduling, and Med-Sync prescription clustering. 

Rather than rewriting the existing TypeScript codebase or attempting to port complex scikit-learn/gradient-boosting models into Node.js, RefillCare deploys as an **independent, isolated Python Microservice Container**. It plugs directly into Mediastra’s AWS infrastructure, communicating with the live Angular frontend and backend via secure REST APIs and sharing the AWS RDS cluster through **strict PostgreSQL Schema Isolation**.

### 1.1 High-Level Microservice Integration Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["1. Client & Ingress Layer"]
        USER["Browser / Pharmacy Staff / Patients"] --> ALB["AWS Application Load Balancer<br/>https://app.mediastra.com"]
    end

    subgraph RoutingLayer["2. ALB Path-Based Routing Rules"]
        ALB -->|"Path: /* (Default)"| ANG["Existing Live Angular SPA<br/>(Client Web App on Port 80/443)"]
        ALB -->|"Path: /api/v1/*"| NODE["Existing Live Backend<br/>(TypeScript / Prism on Port 3000)"]
        ALB -->|"Path: /api/refillcare/*"| FASTAPI["NEW RefillCare REST Service<br/>(FastAPI on Port 8000)"]
        ALB -->|"Path: /ops/refillcare/*"| STREAMLIT["NEW Pharmacist Operations Portal<br/>(Streamlit on Port 8501)"]
    end

    subgraph ServiceInteractions["3. Microservice Communication"]
        NODE -.->|"Internal HTTP (X-API-Key + X-Store-Id)"| FASTAPI
        FASTAPI -->|"WhatsApp Delivery Requests"| XINNO["Xinno WhatsApp Gateway"]
    end

    subgraph DatabaseLayer["4. AWS RDS PostgreSQL 16 Cluster"]
        NODE -->|"Direct Query"| RDS_PUB["public schema<br/>(Live App: users, orders, prescriptions)"]
        FASTAPI -->|"Control Plane"| RDS_CTRL["public schema<br/>(refillcare_stores, refillcare_jobs)"]
        FASTAPI -->|"X-Store-Id: store-apollo-hyd"| RDS_STORE1["store_apollo_hyd schema<br/>(Isolated Sales, Refills, Reminders)"]
        FASTAPI -->|"X-Store-Id: store-care-banjara"| RDS_STORE2["store_care_banjara schema<br/>(Isolated Sales, Refills, Reminders)"]
    end

    classDef alb fill:#ff9900,stroke:#232f3e,stroke-width:2px,color:#fff;
    classDef existing fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff;
    classDef newService fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;
    classDef rds fill:#3b48cc,stroke:#1a237e,stroke-width:2px,color:#fff;
    class ALB alb;
    class ANG,NODE existing;
    class FASTAPI,STREAMLIT,XINNO newService;
    class RDS_PUB,RDS_CTRL,RDS_STORE1,RDS_STORE2 rds;
```

---

## 2. Multi-Tenant Strategy: Store-Level Database Isolation

Healthcare regulations (HIPAA, DISHA, ISO 27001) mandate that patient prescription histories, clinical reminders, and sales records between different pharmacy entities, franchise branches, or partner hospitals must remain **strictly segregated**.

### 2.1 The Two-Tier Isolation Model

RefillCare implements a **Two-Tier Tenancy Model** that balances strict data isolation with zero infrastructure bloat:

```mermaid
flowchart TD
    subgraph RDSInstance["AWS RDS PostgreSQL 16 Instance"]
        subgraph ControlPlane["Control Plane: Shared Orchestration (Schema: public)"]
            STORES["refillcare_stores<br/>• store_id (PK)<br/>• tenant_id<br/>• store_name<br/>• settings (JSON)<br/>• status"]
            JOBS["refillcare_jobs<br/>• job_id (PK)<br/>• store_id (FK)<br/>• kind (retrain/sync)<br/>• status (queued/running)<br/>• dedupe_key"]
            STORES --- JOBS
        end

        subgraph TenantStoreA["Tenant Data Plane: Store A (Schema: store_apollo_hyd)"]
            A_SALES["sales_transactions<br/>(Patient POS sales lines)"]
            A_BATCH["import_batches<br/>(Upload manifests)"]
            A_DEC["refill_decisions<br/>(Forecasted refill cycles)"]
            A_REM["reminder_stages<br/>(6-Stage WhatsApp queue)"]
            A_ML["ml_model_versions<br/>(Store-specific ML registry)"]
            A_BATCH --> A_SALES
            A_SALES --> A_DEC
            A_DEC --> A_REM
            A_ML -.-> A_DEC
        end

        subgraph TenantStoreB["Tenant Data Plane: Store B (Schema: store_care_banjara)"]
            B_SALES["sales_transactions<br/>(Patient POS sales lines)"]
            B_BATCH["import_batches<br/>(Upload manifests)"]
            B_DEC["refill_decisions<br/>(Forecasted refill cycles)"]
            B_REM["reminder_stages<br/>(6-Stage WhatsApp queue)"]
            B_ML["ml_model_versions<br/>(Store-specific ML registry)"]
            B_BATCH --> B_SALES
            B_SALES --> B_DEC
            B_DEC --> B_REM
            B_ML -.-> B_DEC
        end
    end

    FASTAPI_ENGINE["RefillCare Runtime Engine<br/>(SQLAlchemy schema_translate_map)"]
    FASTAPI_ENGINE -->|"Resolves to public"| ControlPlane
    FASTAPI_ENGINE -->|"X-Store-Id: apollo-hyd"| TenantStoreA
    FASTAPI_ENGINE -->|"X-Store-Id: care-banjara"| TenantStoreB

    classDef ctrl fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#1a237e;
    classDef storeA fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20;
    classDef storeB fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100;
    classDef engine fill:#ede7f6,stroke:#673ab7,stroke-width:2px,color:#311b92;
    class ControlPlane,STORES,JOBS ctrl;
    class TenantStoreA,A_SALES,A_BATCH,A_DEC,A_REM,A_ML storeA;
    class TenantStoreB,B_SALES,B_BATCH,B_DEC,B_REM,B_ML storeB;
    class FASTAPI_ENGINE engine;
```

### 2.2 How the Code Enforces Isolation (Runtime Engine)

The isolation mechanism is implemented in `refillcare/tenancy.py` and `database/connection.py`:

1. **Context-Driven Store Resolution:**
   Every API request receives an incoming header `X-Store-Id: <store_id>`. FastAPI middleware parses this header and initializes a thread-safe `ContextVar`:
   ```python
   # refillcare/tenancy.py
   sid = set_current_store(request.headers.get("X-Store-Id"))
   ```
2. **Dynamic PostgreSQL Schema Translation:**
   SQLAlchemy dynamically maps unqualified table names (`sales_transactions`) to the store's dedicated schema using `schema_translate_map`:
   ```python
   # database/connection.py
   eng = engine.execution_options(
       schema_translate_map={None: schema_for_store(sid)}
   )
   ```
   * *Example:* If `store_id="apollo-hyd"`, every query automatically routes to schema `"store_apollo_hyd"`. A query from Store A **can never physically read or write rows from Store B**.

3. **Per-Store Custom Settings Overrides:**
   Different pharmacy branches have unique operational rules. RefillCare stores per-store configuration in `refillcare_stores.settings`, allowing individual branches to configure:
   * `REFILLCARE_PATIENT_IDENTITY`: `"customer_id"` vs `"name_mobile"`.
   * `REFILLCARE_RULE_DOM_ANCHOR`: Day-of-month anchoring (e.g. alignment to the 1st of every month).
   * `REFILLCARE_MAX_DATA_LAG_DAYS`: Maximum allowed gap in POS sales uploads before alert.

---

## 3. Communication Contract: Angular/TypeScript <--> RefillCare

The live Mediastra backend (TypeScript/Node.js) or Angular client interacts with RefillCare via standard JSON HTTP/REST endpoints.

### 3.1 End-to-End Operational Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Staff as Pharmacist / POS System
    participant Angular as Live Angular App
    participant Node as Live Node.js / Prism Backend
    participant RefillCare as RefillCare FastAPI Container
    participant RDS as AWS RDS (PostgreSQL)
    participant Xinno as Xinno WhatsApp Gateway
    actor Patient as Patient (Customer)

    Staff->>Angular: Uploads daily sales file or triggers POS sync
    Angular->>Node: POST /api/v1/sales/sync
    Node->>RefillCare: POST /api/refillcare/v1/sales/ingest<br/>(Headers: X-API-Key, X-Store-Id: store_apollo)
    
    activate RefillCare
    RefillCare->>RDS: Switch schema to 'store_apollo' via schema_translate_map
    RefillCare->>RDS: Insert sales lines into store_apollo.sales_transactions
    RefillCare->>RefillCare: Compute consumption rates & predict next refill date
    RefillCare->>RDS: Save predictions in store_apollo.refill_decisions
    RefillCare->>RDS: Schedule Day -7, -3, -1, 0, +2, +5 stages in store_apollo.reminder_stages
    RefillCare-->>Node: 200 OK (Imported 7,406 records, 3,538 reminders scheduled)
    deactivate RefillCare

    Node-->>Angular: 200 OK (Sync Complete)
    Angular-->>Staff: Display green status & updated dashboard

    Note over RefillCare,RDS: Scheduled Worker triggers due reminders daily at 09:00 AM
    RefillCare->>RDS: Fetch due reminders from store_apollo.reminder_stages (WHERE status='PENDING')
    RefillCare->>Xinno: POST /send-template (Patient Phone, Dosage Regimen, Refill Date)
    Xinno->>Patient: Deliver interactive WhatsApp adherence alert
```

### 3.2 Security & Authentication Contract
Every request sent to RefillCare must include:
* `X-API-Key`: Pre-shared secret generated in AWS Secrets Manager (`REFILLCARE_API_KEY`).
* `X-Store-Id`: Identifier of the specific pharmacy branch (e.g. `store-01`, `apollo-hyd`).
* `X-Tenant-Id`: (Optional) Top-level healthcare group identifier.

### 3.3 Key Integration Endpoints

| Method | Endpoint | Caller | Purpose |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/sales/ingest` | TypeScript / POS Sync | Ingests new POS sales invoices for the designated store. Automatically calculates historical consumption rates and schedules reminders. |
| `GET` | `/api/v1/reminders/due-today` | Angular / Cron Worker | Returns the validated queue of patients due for refill alerts today. |
| `POST` | `/api/v1/whatsapp/dispatch` | Angular Admin / Dispatcher | Triggers delivery of 6-stage WhatsApp reminders via the Xinno Gateway. |
| `GET` | `/api/v1/analytics/kpi-summary` | Angular Dashboard | Returns customer counts, active predictions, and model accuracy statistics. |
| `GET` | `/livez` & `/readyz` | AWS Load Balancer | Health probes for container readiness and zero-downtime rolling deploys. |

---

## 4. DevOps Implementation & Deployment Plan (AWS)

### 4.1 Cloud Infrastructure Architecture

```mermaid
flowchart LR
    subgraph Ingress["AWS Ingress"]
        ALB["AWS Application<br/>Load Balancer"]
    end

    subgraph ECS["AWS ECS Fargate Cluster (mediastra-cluster)"]
        direction TB
        subgraph APITask["Task 1: API Service"]
            API_C["Container: refillcare-api<br/>Port: 8000<br/>Workers: 2-10 (Auto-Scaled)"]
        end

        subgraph WorkerTask["Task 2: Background Worker"]
            WORKER_C["Container: refillcare-worker<br/>Polls: refillcare_jobs<br/>Runs: Retrain & Med-Sync"]
        end

        subgraph StreamlitTask["Task 3: Operations Portal"]
            UI_C["Container: refillcare-ops-ui<br/>Port: 8501<br/>Pharmacist Back-Office UI"]
        end
    end

    subgraph CloudPersistence["AWS Managed Persistence Layer"]
        RDS[("AWS RDS PostgreSQL 16<br/>Multi-AZ Instance")]
        S3[("AWS S3 Bucket<br/>mediastra-refillcare-models-prod")]
        SECRETS["AWS Secrets Manager<br/>(DB URLs, API Keys, Xinno)"]
    end

    ALB -->|"/api/refillcare/*"| API_C
    ALB -->|"/ops/refillcare/*"| UI_C
    
    API_C --> RDS
    WORKER_C --> RDS
    UI_C --> RDS

    API_C -.-> S3
    WORKER_C -.-> S3
    
    SECRETS -.->|Injects Env Vars| ECS

    classDef alb fill:#ff9900,stroke:#232f3e,stroke-width:2px,color:#fff;
    classDef ecs fill:#e8f4f8,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef db fill:#3b48cc,stroke:#1a237e,stroke-width:2px,color:#fff;
    class ALB alb;
    class APITask,WorkerTask,StreamlitTask,API_C,WORKER_C,UI_C ecs;
    class RDS,S3,SECRETS db;
```

---

### 4.2 Phase 1: Infrastructure Provisioning & Database Preparation

1. **Verify RDS PostgreSQL Database:**
   * Ensure PostgreSQL version is 14+ (v16 recommended).
   * Grant schema creation privileges to the application user:
     ```sql
     GRANT ALL PRIVILEGES ON DATABASE mediastra_db TO refillcare_user;
     ALTER USER refillcare_user CREATEDB;
     ```
2. **Execute Multi-Store Idempotent Migration:**
   * Before launching the application containers, run the migration runner task:
     ```bash
     python -m database.migrations --all-stores
     ```
   * This inspects all registered stores in `refillcare_stores`, creates missing PostgreSQL schemas (`CREATE SCHEMA IF NOT EXISTS "store_<id>"`), and executes table versioning without locking existing operational tables.

---

### 4.3 Phase 2: Docker Image Build & ECR Pipeline

The application root contains an enterprise multi-role `Dockerfile` configured with Python 3.13-slim and non-root execution.

**Build and Push Commands:**
```bash
# 1. AWS ECR Login
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com

# 2. Build Container Image
docker build -t mediastra-refillcare:latest .

# 3. Tag and Push
docker tag mediastra-refillcare:latest <AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/mediastra-refillcare:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/mediastra-refillcare:latest
```

---

### 4.4 Phase 3: ECS Task Definition & Environment Configuration

Configure the following environment variables in AWS Systems Manager (SSM) Parameter Store or AWS Secrets Manager:

```ini
# Database (AWS RDS)
DATABASE_URL=postgresql+psycopg://refillcare_user:StrongPassword123@mediastra-rds.c123456789.ap-south-1.rds.amazonaws.com:5432/mediastra_db
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Environment & Tenancy
REFILLCARE_ENV=production
REFILLCARE_LOG_FORMAT=json
REFILLCARE_LOG_LEVEL=INFO
REFILLCARE_STORE_ID=store-default

# Security & API Auth
REFILLCARE_API_KEY=sec_live_9f823bc84e724018a09f87c9
WHATSAPP_WEBHOOK_SECRET=whsec_88492019482019482019

# Cloud Storage & Models (AWS S3)
REFILLCARE_MODEL_BUCKET=mediastra-refillcare-models-prod
REFILLCARE_MODEL_PREFIX=models
REFILLCARE_DATA_DIR=/data/refillcare

# WhatsApp Provider (Xinno Gateway)
XINNO_API_KEY=your_production_xinno_key
XINNO_SENDER_MOBILE=919876543210
XINNO_TEMPLATE_ID=refill_reminder_v1
```

---

### 4.5 Phase 4: Path-Based Load Balancer Routing

Configure the AWS Application Load Balancer (ALB) listener rules:

1. **Rule 1 (RefillCare Public API):**
   * Path: `/api/refillcare/*`
   * Target Group: `refillcare-api-tg` (Port 8000)
   * Health Check: `/livez` (Interval: 15s, Healthy Threshold: 2)

2. **Rule 2 (Pharmacist Back-Office UI):**
   * Path: `/ops/refillcare/*`
   * Target Group: `refillcare-streamlit-tg` (Port 8501)
   * Access: Restrict via IP Allowlist or AWS Cognito / Corporate VPN

3. **Default Rule:**
   * Path: `/*`
   * Target Group: `mediastra-live-angular-tg` (Existing Live App)

---

## 5. Automated CI/CD Pipeline (GitHub Actions Blueprint)

Create `.github/workflows/deploy-refillcare.yml` in your repository:

```yaml
name: Deploy Mediastra RefillCare to AWS ECS

on:
  push:
    branches:
      - main
      - production

jobs:
  test-and-verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
          cache: 'pip'
      - name: Install Dependencies
        run: |
          pip install -r requirements.txt -r requirements-dev.txt
      - name: Run Unit & Regression Test Suite
        run: |
          pytest tests -q --disable-warnings

  build-and-deploy:
    needs: test-and-verify
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and Push Docker Image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: mediastra-refillcare
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Run Database Migrations
        run: |
          aws ecs run-task \
            --cluster mediastra-cluster \
            --task-definition refillcare-migrate \
            --launch-type FARGATE \
            --network-configuration "awsvpcConfiguration={subnets=[${{ secrets.SUBNET_ID }}],securityGroups=[${{ secrets.SG_ID }}],assignPublicIp=ENABLED}"

      - name: Deploy to ECS Services (Zero-Downtime Rolling Update)
        run: |
          aws ecs update-service --cluster mediastra-cluster --service refillcare-api --force-new-deployment
          aws ecs update-service --cluster mediastra-cluster --service refillcare-worker --force-new-deployment
          aws ecs update-service --cluster mediastra-cluster --service refillcare-ops-ui --force-new-deployment
```

---

## 6. Verification & Rollback Procedures

### 6.1 Post-Deployment Smoke Test Checklist
1. **Health Probes:**
   * Query `https://app.mediastra.com/api/refillcare/livez` -> Expect HTTP `200 {"status": "ok"}`.
2. **Schema Verification:**
   * Query RDS: `SELECT schema_name FROM information_schema.schemata;`
   * Confirm schemas exist: `public`, `store_default`, and any registered partner branch schemas.
3. **Tenant Boundary Verification:**
   * Make a test request with `X-Store-Id: store-test-a` and another with `X-Store-Id: store-test-b`.
   * Verify that data inserted under Store A is completely invisible to Store B.
4. **WhatsApp Webhook Audit:**
   * Confirm delivery callback logs in CloudWatch show JSON lines with masked patient phone numbers (`98******45`).

### 6.2 Safe Rollback Strategy
* **Database Level:** RefillCare schema migrations are non-destructive (additive columns only). Rolling back application containers to a previous Docker tag will **never break existing schema state**.
* **Container Level:** If any defect is detected, AWS ECS instantly rolls back to the previous stable Task Definition with zero downtime (`aws ecs update-service --service refillcare-api --task-definition refillcare-api:<PREVIOUS_VERSION>`).

---

## 7. Action Items for the Engineering Team

| Team Member | Deliverable | Target Timeline |
| :--- | :--- | :--- |
| **DevOps / Cloud Lead** | Create ECR repository, S3 model bucket, and ECS Task Definitions. Configure ALB path routing. | Day 1–2 |
| **DBA / Database Lead** | Create `refillcare_user` in AWS RDS and execute initial migration (`python -m database.migrations --all-stores`). | Day 2 |
| **Backend Lead (TypeScript)** | Add HTTP client helper in existing Node.js service to forward POS sales lines to `POST /api/refillcare/v1/sales/ingest`. | Day 3–4 |
| **Frontend Lead (Angular)** | Add iframe or navigation tab pointing to RefillCare Operations Portal for pharmacy staff. | Day 4–5 |
| **QA / Clinical Ops** | Run pilot tests on test store schema (`store_qa_pilot`), verify WhatsApp reminder formatting and Med-Sync bundles. | Day 6 |
