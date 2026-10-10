# FlipPrice AI — Production Deployment Guide

This guide covers deploying the **FlipPrice AI Decision Engine & Shopify App** to production or staging.

---

## Architecture Overview
The system consists of two deployable components:
1. **Decision Engine & Webhook API (Backend):**
   - FastAPI REST service running on port `8000`.
   - Persists multi-tenant audit logs & OAuth sessions in SQLite (`data/tenants.db`).
   - Containerized via standard `Dockerfile` and `docker-compose.yml`.
2. **Landing Page & Bleed Calculator (Frontend):**
   - Standalone responsive web client in [`docs/commercial/landing.html`](file:///c:/Users/vansh/OneDrive/Desktop/rto-shield-prep/rto-shield/docs/commercial/landing.html).
   - Zero-dependency static HTML/CSS/JS (deployable on Vercel, Cloudflare Pages, or Netlify).

---

## Method 1: Cloud PaaS Deployment (Recommended — Render / Railway)

### Deploying to Render.com (1-Click Blueprint)
1. Push your repository to GitHub.
2. Sign in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint** and select your GitHub repository.
4. Render detects [`render.yaml`](file:///c:/Users/vansh/OneDrive/Desktop/rto-shield-prep/rto-shield/render.yaml) automatically:
   - **Region:** Singapore (`sin`) — provides lowest network latency (~30–45ms) to Indian merchant checkouts.
   - **Persistent Disk:** Mounted at `/app/data` (preserves tenant DB across redeploys).
   - **Health Check:** `/health`.
5. Under Environment Variables in Render Dashboard, configure:
   - `SHOPIFY_API_SECRET`: Your Shopify App Secret (from Shopify Partner Dashboard).
   - `SHOPIFY_API_KEY`: Your Shopify App Client ID.
6. Click **Apply**. Once built, you will receive a public HTTPS URL:
   `https://flipprice-api.onrender.com`

---

## Method 2: Testing Live with Shopify Test Store (Local Tunnel via Ngrok)

To test live webhooks from a Shopify development store before pushing to the cloud:

1. **Start the API locally:**
   ```bash
   python -m uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Expose your local port via ngrok:**
   ```bash
   ngrok http 8000
   ```
   *Example output URL:* `https://abc123xyz.ngrok-free.app`

3. **Configure Shopify Webhook:**
   - In your Shopify Partner Dashboard $\rightarrow$ **App Setup** $\rightarrow$ **App URL**: `https://abc123xyz.ngrok-free.app`
   - Allowed redirection URL: `https://abc123xyz.ngrok-free.app/shopify/auth/callback`
   - Webhook URL for `orders/create`: `https://abc123xyz.ngrok-free.app/shopify/webhooks/orders/create`

4. **Fire a test order:**
   Run our built-in simulator to verify:
   ```bash
   python scripts/commercial/simulate_live_shopify_order.py
   ```

---

## Method 3: Self-Hosted Production VM (AWS EC2 / DigitalOcean / Hetzner)

For high-volume merchants (>100k orders/month) requiring dedicated hardware:

### 1. Provision a Linux Server
- Recommended: Ubuntu 24.04 LTS (2 vCPU, 4GB RAM).
- Select a region close to India (AWS `ap-south-1` Mumbai or DigitalOcean `blr1` Bengaluru).

### 2. Clone & Launch with Docker Compose
```bash
# Install Docker & Docker Compose
curl -fsSL https://get.docker.com | sh

# Clone repo & enter
git clone https://github.com/mahirbhat70-eng/rto-shield.git
cd rto-shield

# Build and start container in background
docker compose up -d --build
```

### 3. Setup Nginx Reverse Proxy with SSL (Certbot)
Create `/etc/nginx/sites-available/flipprice`:
```nginx
server {
    server_name api.flipprice.ai;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site and acquire free SSL certificate:
```bash
sudo ln -s /etc/nginx/sites-available/flipprice /etc/nginx/sites-enabled/
sudo certbot --nginx -d api.flipprice.ai
```

---

## Method 4: Deploying the Landing Page & Bleed Calculator

The landing page ([`docs/commercial/landing.html`](file:///c:/Users/vansh/OneDrive/Desktop/rto-shield-prep/rto-shield/docs/commercial/landing.html)) is zero-dependency static HTML:

### Deploy to Vercel (Fastest):
1. Rename or copy `docs/commercial/landing.html` to `index.html` in a separate frontend branch or public repo.
2. Run:
   ```bash
   npx vercel deploy --prod
   ```
3. Custom domain: Point `flipprice.ai` CNAME to `cname.vercel-dns.com`.

### Deploy to Cloudflare Pages:
1. In Cloudflare Dashboard $\rightarrow$ **Pages** $\rightarrow$ **Create a project**.
2. Connect your repository, set build directory to `docs/commercial`.
3. Deploy $\rightarrow$ Instant global CDN with free SSL.

---

## Post-Deployment Verification & Smoke Test

Once deployed, verify the endpoint using `curl`:

### 1. Health & Model Digest Check
```bash
curl -X GET https://YOUR_API_DOMAIN/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "FlipPrice AI Decision Engine",
  "version": "2.0.0",
  "primary_model": "logistic_regression",
  "active_policy": "constants_p1"
}
```

### 2. Live Scoring Check
```bash
curl -X POST https://YOUR_API_DOMAIN/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fp_live_demo_merchant_123" \
  -d '{
    "order_value": 1499.0,
    "pincode": "110001",
    "category": "Apparel",
    "payment_method": "COD"
  }'
```
**Expected Response:**
```json
{
  "recommended_action": "ALLOW_COD",
  "probability_rto": 0.172,
  "friction_inr": 0.0,
  "top_risk_drivers": [...]
}
```

---

## Production Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `PORT` | Optional | `8000` | Port listened by uvicorn. |
| `RTO_SHIELD_ENV` | Optional | `development` | Set to `production` in live deployments. |
| `FLIPPRICE_PII_SALT` | Optional | `rto_shield_dpdp_2026_salt` | Salt used for DPDP Act 2023 customer hashing. |
| `SHOPIFY_API_KEY` | For Shopify OAuth | `—` | Client ID from Shopify Partner Dashboard. |
| `SHOPIFY_API_SECRET` | For Webhook Auth | `—` | Shared secret for verifying `X-Shopify-Hmac-Sha256`. |
