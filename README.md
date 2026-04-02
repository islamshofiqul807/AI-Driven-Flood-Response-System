# AI-Driven-Flood-Response-System

# 🌍 Disaster Response System

A simulation-based disaster response system with an interactive dashboard and scenario analysis tools. This project helps visualize and evaluate disaster scenarios using data-driven approaches.

---

## 🚀 Features

- 📊 Interactive dashboard using Streamlit  
- 🗺️ Map visualization with Folium  
- 📈 Data analysis with Pandas, NumPy, SciPy  
- 🔗 Network modeling using NetworkX   
- 📦 Scenario-based simulation (S1, S2, S3, etc.)  
- ⚙️ CLI and dashboard execution modes  

---

## 📁 Project Setup

### 🔹 Step 1 — Extract the Project

```bash
tar -xzf disaster-response-system-v3-final.tar.gz
cd disaster-response-system

python -m venv venv

venv\Scripts\activate

pip install streamlit folium streamlit-folium plotly pandas numpy scipy shapely networkx pydantic python-dotenv geojson matplotlib

python -m streamlit run dashboard/app.py

python run_demo.py --all
