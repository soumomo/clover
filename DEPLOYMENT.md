# Clover Deployment & Hosting Guide

This guide details all methods to host and run the **Clover** canopy delineation & carbon MRV platform.

---

## 1. Live Production Cloud (Official)

Clover is hosted and running 24/7 on Streamlit Community Cloud with automatic continuous deployment from GitHub:

> 🌐 **Live URL:** **[https://tryclover.streamlit.app](https://tryclover.streamlit.app)**

* **Uptime**: 24/7 high-availability cloud container.
* **Security**: Enforced TLS/SSL HTTPS.
* **Auto-Sync**: Pushing to branch `main` automatically hot-reloads the production app.

---

## 2. Local High-Performance Execution with `uv`

Clover uses `uv` for sub-second dependency resolution, deterministic builds via `uv.lock`, and isolated execution:

```bash
# Clone the repository
git clone https://github.com/soumomo/clover.git
cd clover

# Run directly in 1 command (uv auto-provisions the environment)
uv run streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> [!TIP]
> To deterministically install exact locked dependencies from `uv.lock`:
> ```bash
> uv sync
> uv run streamlit run app.py
> ```

---
## 3. Containerized Deployment (Docker / Cloud Run / Railway)

The included `Dockerfile` builds a production-ready Debian image with GDAL and OpenCV.

### Local Docker Build & Run:
```bash
docker build -t clover-app .
docker run -p 8501:8501 clover-app
```

### Deploy to Render or Railway:
1. Connect your GitHub repository to Render / Railway.
2. Choose **Docker** as the environment.
3. Expose port `8501`.
4. Deploy!
