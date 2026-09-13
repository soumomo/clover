# Clover Deployment & Hosting Guide

This guide details all methods to host and deploy the **Clover** canopy delineation & carbon MRV platform.

---

## 1. Instant Public Access (Live Now)

A Cloudflare Tunnel is running locally, providing a global HTTPS endpoint with zero configuration:

> 🌐 **Live URL:** [https://meal-society-described-incident.trycloudflare.com](https://meal-society-described-incident.trycloudflare.com)

- **SSL:** Enabled (Cloudflare edge certificate).
- **WebSockets:** Full real-time support for Streamlit widgets and Folium maps.
- **Access:** Can be opened directly from any mobile device, tablet, or external network.

To start or restart the tunnel anytime:
```bash
cloudflared tunnel --url http://localhost:8501
```

---

## 2. Streamlit Community Cloud (Recommended for Hackathons)

Streamlit Community Cloud offers free, permanent hosting directly connected to your GitHub repository.

### Prerequisites in this repo:
- `app.py` (Main entry point)
- `requirements.txt` (Python packages)
- `packages.txt` (Debian system libraries: GDAL, Mesa GL)
- `.streamlit/config.toml` (Theme & server settings)

### Steps to Deploy:
1. **Initialize Git and Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: Clover AI canopy delineation platform"
   git branch -M main
   # Create a repo on github.com, then:
   git remote add origin https://github.com/<your-username>/clover-canopy-tool.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository: `<your-username>/clover-canopy-tool`.
   - Set **Branch**: `main`.
   - Set **Main file path**: `app.py`.
   - Click **"Deploy!"**.

Streamlit Cloud will automatically read `packages.txt` to install GDAL and `requirements.txt` to install the DeepForest/PyTorch stack.

---

## 3. Hugging Face Spaces (Free Cloud Hosting)

Hugging Face Spaces supports Streamlit apps out of the box.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Space name: `clover-carbon-mrv`.
3. Select **Streamlit** as the Space SDK.
4. Set hardware: **CPU basic (free)** or **T4 GPU**.
5. Push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/clover-carbon-mrv
   git push hf main
   ```

---

## 4. Containerized Deployment (Render, Railway, GCP Cloud Run, AWS ECS)

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
