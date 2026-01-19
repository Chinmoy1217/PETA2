# ETA Insight - Quick Start Guide

## Prerequisites
- Python 3.10 virtual environment activated
- Node.js and npm installed
- All dependencies installed

---

## 🚀 Starting the Application

### Option 1: Two Separate Terminals (Recommended)

#### Terminal 1 - Backend (FastAPI)
```powershell
# Navigate to project root
cd c:\Users\Administrator\.gemini\antigravity\PETA2

# Activate virtual environment (if not already activated)
.\venv310\Scripts\Activate.ps1

# Start backend server
.\venv310\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Backend URLs:**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

#### Terminal 2 - Frontend (Vite + React)
```powershell
# Navigate to frontend directory
cd c:\Users\Administrator\.gemini\antigravity\PETA2\frontend

# Start frontend dev server
npm run dev -- --host
```

**Expected Output:**
```
  VITE v7.3.1  ready in 246 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://10.0.3.8:5173/
  ➜  press h + enter to show help
```

**Frontend URL:**
- Application: http://localhost:5173

---

### Option 2: Single Terminal (Background Processes)

**Not recommended for development** - harder to see errors/logs

```powershell
# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\Users\Administrator\.gemini\antigravity\PETA2; .\venv310\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

# Start frontend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd c:\Users\Administrator\.gemini\antigravity\PETA2\frontend; npm run dev -- --host"
```

---

## 🛑 Stopping the Servers

### In Terminal
- Press `Ctrl + C` in each terminal window

### Kill Processes (if hung)
```powershell
# Find and kill backend
Get-Process -Name python | Where-Object {$_.Path -like "*venv310*"} | Stop-Process -Force

# Find and kill frontend
Get-Process -Name node | Stop-Process -Force
```

---

## 🔍 Verify Services are Running

### Check Backend
```powershell
curl http://localhost:8000/metrics
```
Should return JSON with metrics.

### Check Frontend
Open browser: http://localhost:5173

---

## 📝 Common Commands

### Backend
```powershell
# Install dependencies
.\venv310\Scripts\pip install -r requirements.txt

# Run without reload (production-like)
.\venv310\Scripts\uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Run on different port
.\venv310\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

### Frontend
```powershell
# Install dependencies
npm install

# Run dev server (localhost only)
npm run dev

# Run dev server (network accessible)
npm run dev -- --host

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🐛 Troubleshooting

### Backend won't start
**Error: "Address already in use"**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

**Error: "ModuleNotFoundError"**
```powershell
# Reinstall dependencies
.\venv310\Scripts\pip install -r requirements.txt
```

### Frontend won't start
**Error: "Port 5173 is already in use"**
```powershell
# Kill node processes
Get-Process -Name node | Stop-Process -Force

# Or use different port
npm run dev -- --port 5174 --host
```

**Error: "Module not found"**
```powershell
# Reinstall dependencies
npm install
```

---

## 🌐 Access from Another Device

**Backend**: http://YOUR_IP:8000  
**Frontend**: http://YOUR_IP:5173

Find your IP:
```powershell
ipconfig | findstr IPv4
```

---

## 📌 Quick Reference

| Service | Command | URL |
|:--------|:--------|:----|
| **Backend** | `.\venv310\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000` | http://localhost:8000 |
| **Frontend** | `npm run dev -- --host` | http://localhost:5173 |
| **Swagger** | (Auto-available with backend) | http://localhost:8000/docs |

---

## 💡 Pro Tips

1. **Always start backend FIRST**, then frontend
2. **Keep both terminals open** to see real-time logs
3. **Use Swagger** (`/docs`) if frontend has issues
4. **Frontend auto-reloads** on file changes
5. **Backend auto-reloads** on `.py` file changes (with `--reload` flag)
