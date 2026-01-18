import { useState, useEffect } from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, LineChart, Line, AreaChart, Area
} from 'recharts';
import { LayoutDashboard, UploadCloud, Calculator, Activity, Truck, CheckCircle, TrendingUp, AlertTriangle, Download, LogOut, Lock, Database, Webhook } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Fix Leaflet Icon Issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [metrics, setMetrics] = useState(null);
  const [plots, setPlots] = useState(null);
  const [theme, setTheme] = useState('dark');
  const API_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

  useEffect(() => {
    document.body.className = theme;
    if (isAuthenticated) {
      fetch(`${API_URL}/metrics`).then(res => res.json()).then(setMetrics);
      fetch(`${API_URL}/plots`).then(res => res.json()).then(setPlots);
    }
  }, [isAuthenticated, theme]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

  const handleLogin = (status) => {
    if (status) {
      setIsAuthenticated(true);
      setActiveTab('upload');
    }
  }

  if (!isAuthenticated) {
    return <LoginView onLogin={handleLogin} API_URL={API_URL} theme={theme} toggleTheme={toggleTheme} />;
  }

  return (
    <div className={`app-container ${theme}`} style={{ background: theme === 'dark' ? '#0f172a' : '#f8fafc', color: theme === 'dark' ? '#fff' : '#1e293b', minHeight: '100vh', display: 'flex' }}>
      <nav className={`sidebar ${theme}`} style={{ width: '260px', padding: '2rem', display: 'flex', flexDirection: 'column', background: theme === 'dark' ? '#1e293b' : '#fff', borderRight: theme === 'dark' ? '1px solid #334155' : '1px solid #e2e8f0' }}>
        <div className="brand" style={{ marginBottom: '3rem', display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '1.2rem', fontWeight: 'bold' }}>
          <Truck className="brand-icon" size={32} color={theme === 'dark' ? '#60a5fa' : '#2563eb'} />
          <span style={{ color: theme === 'dark' ? '#fff' : '#1e293b' }}>ETA Insight</span>
        </div>
        <ul className="nav-links" style={{ listStyle: 'none', padding: 0 }}>
          {['upload', 'dashboard', 'tracking', 'simulator'].map(tab => (
            <li
              key={tab}
              className={`nav-item ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
              style={{ padding: '1rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1rem', color: activeTab === tab ? (theme === 'dark' ? '#fff' : '#2563eb') : (theme === 'dark' ? '#94a3b8' : '#64748b'), background: activeTab === tab ? (theme === 'dark' ? 'rgba(59, 130, 246, 0.2)' : '#e0e7ff') : 'transparent', borderRadius: '0.5rem', marginBottom: '0.5rem' }}
            >
              {tab === 'upload' && <UploadCloud size={20} />}
              {tab === 'dashboard' && <LayoutDashboard size={20} />}
              {tab === 'tracking' && <CheckCircle size={20} />}
              {tab === 'simulator' && <Calculator size={20} />}
              {tab.charAt(0).toUpperCase() + tab.slice(1).replace('upload', 'Data Ingestion')}
            </li>
          ))}
        </ul>

        <div style={{ marginTop: 'auto' }}>
          <button
            className="nav-item"
            onClick={toggleTheme}
            style={{ width: '100%', border: 'none', background: 'none', color: theme === 'dark' ? '#94a3b8' : '#64748b', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1rem', padding: '10px', marginBottom: '0.5rem' }}
          >
            {theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
          </button>
          <button className="nav-item" onClick={() => setIsAuthenticated(false)} style={{ width: '100%', border: 'none', background: 'none', color: theme === 'dark' ? '#94a3b8' : '#64748b', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1rem', padding: '10px' }}>
            <LogOut size={20} /> Logout
          </button>
        </div>
      </nav>

      <main className="main-content" style={{ flex: 1, padding: '2rem', overflowY: 'auto', background: theme === 'dark' ? 'transparent' : '#f1f5f9' }}>
        <header className="header-bar">
          <h1 className="page-title" style={{ color: theme === 'dark' ? '#fff' : '#1e293b' }}>
            {activeTab === 'dashboard' && 'Operations Overview'}
            {activeTab === 'tracking' && 'Real-Time Tracking'}
            {activeTab === 'simulator' && 'Instant ETA Calculator'}
            {activeTab === 'upload' && 'Data Ingestion Hub'}
          </h1>
          <div className="model-badge" style={{ background: theme === 'dark' ? 'rgba(16, 185, 129, 0.2)' : '#dcfce7', color: '#10b981' }}>
            <span className="status-dot"></span>
            Model Active (XGBoost v1.0)
          </div>
        </header>

        {activeTab === 'dashboard' && <DashboardView metrics={metrics} plots={plots} />}
        {activeTab === 'tracking' && <TrackingView API_URL={API_URL} />}
        {activeTab === 'simulator' && <SimulatorView API_URL={API_URL} />}
        {activeTab === 'upload' && <UploadView API_URL={API_URL} theme={theme} />}
      </main>
    </div>
  );
}

function LoginView({ onLogin, API_URL, theme, toggleTheme }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSignup, setShowSignup] = useState(false);

  // Video Loop workaround for precise 1.5s - 25s loop
  const [videoKey, setVideoKey] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setVideoKey(prev => prev + 1);
    }, 23500); // 25s - 1.5s = 23.5s
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (res.ok) {
        onLogin(true);
      } else {
        setError("Authentication failed.");
      }
    } catch (err) {
      setError("Unable to connect to service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100vw', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {/* Background Video - YouTube Embed */}
      <div style={{
        position: 'absolute',
        top: '-10%', left: '-10%',
        width: '120%', height: '120%',
        zIndex: -1,
        filter: theme === 'dark' ? 'brightness(0.7)' : 'brightness(0.8) contrast(1.1)'
      }}>
        <iframe
          key={videoKey}
          width="100%"
          height="100%"
          src={`https://www.youtube.com/embed/5ye6OEhLFZc?autoplay=1&mute=1&controls=0&loop=1&playlist=5ye6OEhLFZc&showinfo=0&rel=0&iv_load_policy=3&fs=0&disablekb=1&start=2&end=25`}
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          style={{ pointerEvents: 'none', objectFit: 'cover' }}
        />
      </div>

      <div style={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <button onClick={toggleTheme} style={{ background: 'rgba(255,255,255,0.2)', backdropFilter: 'blur(5px)', border: 'none', padding: '0.5rem 1rem', borderRadius: '20px', color: '#fff', cursor: 'pointer' }}>
          {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>

      <div className="glass-card" style={{ width: '450px', padding: '3rem', textAlign: 'center', background: theme === 'dark' ? 'rgba(15, 23, 42, 0.75)' : 'rgba(255, 255, 255, 0.85)', backdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.3)', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}>
        <div style={{ marginBottom: '2.5rem' }}>
          <div style={{ background: 'rgba(59, 130, 246, 0.2)', padding: '1.2rem', borderRadius: '50%', display: 'inline-flex', marginBottom: '1.5rem', boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' }}>
            <Truck size={40} color="#60a5fa" />
          </div>
          <h1 style={{ color: theme === 'dark' ? '#fff' : '#1e293b', margin: '0 0 0.5rem', fontSize: '2rem', fontWeight: '300', letterSpacing: '1px', lineHeight: '1.3' }}>
            Welcome to DataTalks<br />ETA Insights
          </h1>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1.2rem' }}>
            <input
              type="text"
              className="glass-input"
              style={{ width: '100%', padding: '1rem', fontSize: '1rem', background: theme === 'dark' ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.5)', borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#cbd5e1', color: theme === 'dark' ? '#fff' : '#334155', boxSizing: 'border-box' }}
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Username"
            />
          </div>
          <div style={{ marginBottom: '2rem' }}>
            <input
              type="password"
              className="glass-input"
              style={{ width: '100%', padding: '1rem', fontSize: '1rem', background: theme === 'dark' ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.5)', borderColor: theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#cbd5e1', color: theme === 'dark' ? '#fff' : '#334155', boxSizing: 'border-box' }}
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Password"
            />
          </div>

          {error && <div className="error-banner" style={{ marginBottom: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', color: '#ef4444' }}>{error}</div>}

          <button type="submit" className="action-btn" style={{ width: '100%', padding: '1rem', fontSize: '1.1rem', letterSpacing: '0.5px' }} disabled={loading}>
            {loading ? 'Authenticating...' : 'Access Portal'}
          </button>
        </form>

        <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: `1px solid ${theme === 'dark' ? 'rgba(255,255,255,0.1)' : '#cbd5e1'}` }}>
          <p style={{ color: theme === 'dark' ? '#94a3b8' : '#64748b', fontSize: '0.9rem', marginBottom: '1rem' }}>New to the platform?</p>
          <style>
            {`
              .create-acc-btn {
                width: 100%;
                padding: 0.8rem;
                background: transparent;
                border: 1px solid ${theme === 'dark' ? '#475569' : '#94a3b8'};
                color: ${theme === 'dark' ? '#94a3b8' : '#64748b'};
                cursor: pointer;
                transition: all 0.3s ease;
                border-radius: 0.5rem;
              }
              .create-acc-btn:hover {
                border-color: #3b82f6;
                color: #60a5fa;
                background: rgba(59, 130, 246, 0.1);
              }
            `}
          </style>
          <button
            className="create-acc-btn"
            onClick={() => setShowSignup(true)}
          >
            Create Account
          </button>
        </div>
      </div>
      {showSignup && <SignupModal onClose={() => setShowSignup(false)} API_URL={API_URL} theme={theme} />}
    </div>
  );
}

function SignupModal({ onClose, API_URL }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [msg, setMsg] = useState(null);
  const [error, setError] = useState(null);

  const handleSignup = async (e) => {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setError(null);
    try {
      const res = await fetch(`${API_URL}/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (res.ok) {
        setMsg("Account created! Please login.");
        setTimeout(() => onClose(), 2000); // Close after success
      } else {
        setError("Registration failed.");
      }
    } catch (err) {
      setError("Service unavailable.");
    }
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="glass-card" style={{ width: '350px', padding: '2rem', background: '#1e293b', border: '1px solid #3b82f6' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff' }}>Sign Up</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '1.5rem' }}>&times;</button>
        </div>

        {msg ? <div style={{ color: '#10b981', textAlign: 'center', padding: '2rem 0' }}>{msg}</div> : (
          <form onSubmit={handleSignup}>
            <div style={{ marginBottom: '1rem' }}>
              <input type="text" placeholder="Choose Username" value={username} onChange={e => setUsername(e.target.value)} className="glass-input" style={{ width: '100%', boxSizing: 'border-box' }} required />
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="glass-input" style={{ width: '100%', boxSizing: 'border-box' }} required />
            </div>
            <div style={{ marginBottom: '1.5rem' }}>
              <input type="password" placeholder="Confirm Password" value={confirm} onChange={e => setConfirm(e.target.value)} className="glass-input" style={{ width: '100%', boxSsizing: 'border-box' }} required />
            </div>
            {error && <div style={{ color: '#ef4444', marginBottom: '1rem', fontSize: '0.9rem' }}>{error}</div>}
            <button type="submit" className="action-btn" style={{ width: '100%' }}>Register</button>
          </form>
        )}
      </div>
    </div>
  );
}

function DashboardView({ metrics, plots }) {
  const [comparison, setComparison] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/comparison").then(res => res.json()).then(setComparison);
  }, []);

  if (!metrics) return <div style={{ padding: '2rem' }}>Loading Analytic Engine...</div>;

  return (
    <>
      <div className="card-grid">
        <div className="glass-card">
          <div className="stat-header">Best Accuracy</div>
          <div className="stat-value" style={{ color: '#10b981' }}>{metrics.accuracy}%</div>
          <div className="sub-text">{metrics.name}</div>
        </div>
        <div className="glass-card">
          <div className="stat-header">Avg Error (RMSE)</div>
          <div className="stat-value">{metrics.rmse}h</div>
        </div>
        <div className="glass-card">
          <div className="stat-header">Total Shipments</div>
          <div className="stat-value">600k+</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="glass-card chart-container">
          <div className="stat-header">Model Battle (Accuracy %)</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparison} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} />
              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6' }} />
              <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}>
                {comparison.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={
                    entry.name === 'XGBoost' ? '#10b981' :
                      entry.name === 'Random Forest' ? '#8b5cf6' :
                        '#64748b'
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card chart-container">
          <div className="stat-header">Mode-Specific Performance</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={plots?.mode_performance || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="mode" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[60, 100]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6' }} />
              <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                {plots?.mode_performance?.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={
                    entry.mode.toUpperCase() === 'AIR' ? '#3b82f6' :
                      entry.mode.toUpperCase() === 'RAIL' ? '#f59e0b' :
                        '#10b981'
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div className="glass-card chart-container">
          <div className="stat-header">Variance Map (Predicted vs Actual)</div>
          {!plots?.scatter_data ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Loading Data...</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart key={plots.scatter_data.length}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" dataKey="Actual" name="Actual" stroke="#94a3b8" unit="h" />
                <YAxis type="number" dataKey="Pred" name="Predicted" stroke="#94a3b8" unit="h" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6' }} />
                <Scatter name="Shipments" data={plots.scatter_data} fill="#3b82f6" />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="glass-card chart-container">
          <div className="stat-header">Operations Volume Trend</div>
          {!plots?.timeline_data ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Loading Data...</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart key={plots.timeline_data.length} data={plots.timeline_data}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#10b981' }} />
                <Area type="monotone" dataKey="count" stroke="#10b981" fillOpacity={1} fill="url(#colorCount)" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </>
  );
}

// Internal component for map logic that requires 'useMap' context
function MapEffects({ src, dest, mode, waypoints }) {
  const map = useMap();
  const [routeCoords, setRouteCoords] = useState(null);

  // 1. Fit Bounds (Auto-Zoom)
  useEffect(() => {
    if (src && dest) {
      const bounds = [[src.lat, src.lon], [dest.lat, dest.lon]];
      if (waypoints && waypoints.length > 0) {
        waypoints.forEach(wp => bounds.push([wp.lat, wp.lon]));
      }
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [src, dest, map, waypoints]);

  // 2. Routing Logic
  function decodePolyline(str, precision) {
    var index = 0, lat = 0, lng = 0, coordinates = [], shift = 0, result = 0, byte = null, latitude_change, longitude_change, factor = Math.pow(10, precision || 5);
    while (index < str.length) {
      byte = null; shift = 0; result = 0;
      do { byte = str.charCodeAt(index++) - 63; result |= (byte & 0x1f) << shift; shift += 5; } while (byte >= 0x20);
      latitude_change = (result & 1 ? ~(result >> 1) : result >> 1);
      shift = result = 0;
      do { byte = str.charCodeAt(index++) - 63; result |= (byte & 0x1f) << shift; shift += 5; } while (byte >= 0x20);
      longitude_change = (result & 1 ? ~(result >> 1) : result >> 1);
      lat += latitude_change; lng += longitude_change;
      coordinates.push([lat / factor, lng / factor]);
    }
    return coordinates;
  }

  function getGreatCircle(start, end, numPoints = 100) {
    const toRad = (d) => d * Math.PI / 180;
    const toDeg = (r) => r * 180 / Math.PI;
    const lat1 = toRad(start[0]), lon1 = toRad(start[1]);
    const lat2 = toRad(end[0]), lon2 = toRad(end[1]);
    const d = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin((lat1 - lat2) / 2), 2) + Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin((lon1 - lon2) / 2), 2)));
    let points = [];
    for (let i = 0; i <= numPoints; i++) {
      const f = i / numPoints;
      const A = Math.sin((1 - f) * d) / Math.sin(d);
      const B = Math.sin(f * d) / Math.sin(d);
      const x = A * Math.cos(lat1) * Math.cos(lon1) + B * Math.cos(lat2) * Math.cos(lon2);
      const y = A * Math.cos(lat1) * Math.sin(lon1) + B * Math.cos(lat2) * Math.sin(lon2);
      const z = A * Math.sin(lat1) + B * Math.sin(lat2);
      const lat = toDeg(Math.atan2(z, Math.sqrt(x * x + y * y)));
      const lon = toDeg(Math.atan2(y, x));
      points.push([lat, lon]);
    }
    return points;
  }

  useEffect(() => {
    if (!src || !dest) return;
    setRouteCoords(null);
    const fetchRoute = async () => {
      const m = mode ? mode.toUpperCase() : 'UNKNOWN';
      if (['ROAD', 'TRUCK', 'RAIL'].includes(m)) {
        try {
          // OSRM does not easily support multi-stop via simple API for free, so we stick to point-to-point or ignore waypoints for MVP Road
          const url = `http://router.project-osrm.org/route/v1/driving/${src.lon},${src.lat};${dest.lon},${dest.lat}?overview=full`;
          const res = await fetch(url);
          const data = await res.json();
          if (data.code === 'Ok' && data.routes?.[0]) setRouteCoords(decodePolyline(data.routes[0].geometry, 5));
        } catch (err) { console.warn("OSRM Failed", err); }
      } else if (['AIR', 'OCEAN', 'SEA'].includes(m)) {
        if (waypoints && waypoints.length > 0) {
          // Segmented Great Circles
          let allPoints = [];
          let current = [src.lat, src.lon];

          waypoints.forEach(wp => {
            const seg = getGreatCircle(current, [wp.lat, wp.lon]);
            allPoints = [...allPoints, ...seg];
            current = [wp.lat, wp.lon];
          });
          // Final leg
          const lastLeg = getGreatCircle(current, [dest.lat, dest.lon]);
          allPoints = [...allPoints, ...lastLeg];
          setRouteCoords(allPoints);
        } else {
          setRouteCoords(getGreatCircle([src.lat, src.lon], [dest.lat, dest.lon]));
        }
      }
    };
    fetchRoute();
  }, [src, dest, mode, waypoints]);

  const getPathColor = () => {
    const m = mode ? mode.toUpperCase() : '';
    if (m === 'RAIL') return '#ef4444';
    if (m === 'ROAD' || m === 'TRUCK') return '#10b981';
    if (m === 'OCEAN' || m === 'SEA') return '#3b82f6';
    return '#8b5cf6';
  };
  const getDashArray = () => {
    const m = mode ? mode.toUpperCase() : '';
    if (m === 'RAIL') return "20, 20";
    if (m === 'OCEAN' || m === 'SEA') return "5, 10";
    return null;
  };

  return (
    <>
      <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <Marker position={[src.lat, src.lon]}><Popup>POL: {src.code}</Popup></Marker>
      <Marker position={[dest.lat, dest.lon]}><Popup>POD: {dest.code}</Popup></Marker>
      {routeCoords && <Polyline positions={routeCoords} color={getPathColor()} dashArray={getDashArray()} weight={4} opacity={0.8} />}
      {mode === 'RAIL' && routeCoords && <Polyline positions={routeCoords} color="white" dashArray="20, 20" dashOffset="10" weight={2} />}
    </>
  );
}

// Main shell component
function RealWorldMap({ src, dest, mode, waypoints }) {
  return (
    <div style={{ height: '300px', width: '100%', borderRadius: '0.5rem', overflow: 'hidden', border: '1px solid #334155' }}>
      <MapContainer center={[20, 0]} zoom={2} style={{ height: '100%', width: '100%' }}>
        <MapEffects src={src} dest={dest} mode={mode} waypoints={waypoints} />
      </MapContainer>
    </div>
  );
}


function SimulatorView({ API_URL }) {
  const [formData, setFormData] = useState({ PolCode: 'USLAX', PodCode: 'CNSHA', ModeOfTransport: 'Ocean' }); // Default to Ocean
  const [locations, setLocations] = useState([]);
  const [scenario, setScenario] = useState({ congestion: 0, weather: 0 });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/locations`)
      .then(res => res.json())
      .then(data => {
        setLocations(data);
        // Ensure defaults are valid or pick first
        if (data.length > 0) {
          // Keep defaults if valid, else update
          setFormData(prev => ({
            ...prev,
            PolCode: data.includes(prev.PolCode) ? prev.PolCode : data[0],
            PodCode: data.includes(prev.PodCode) ? prev.PodCode : data[1] || data[0]
          }));
        }
      })
      .catch(err => console.error("Failed to load locations", err));
  }, [API_URL]);

  const handleSimulate = async () => {
    setError(null);
    setResult(null);
    const payload = {
      ...formData,
      congestion_level: parseInt(scenario.congestion),
      weather_severity: parseInt(scenario.weather)
    };

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          pol: formData.PolCode,
          pod: formData.PodCode,
          mode: formData.ModeOfTransport,
          train: 'false'
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Simulation Failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  };

  const simResult = result?.predictions?.[0];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2fr', gap: '2rem', alignItems: 'start' }}>
      <div className="glass-card">
        <h3 style={{ marginTop: 0 }}>Trip Parameters</h3>
        <div className="form-group">
          <label>Origin Port (POL)</label>
          <select
            value={formData.PolCode}
            onChange={e => setFormData({ ...formData, PolCode: e.target.value })}
            className="glass-input"
          >
            {locations.map(loc => <option key={loc} value={loc}>{loc}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label>Destination Port (POD)</label>
          <select
            value={formData.PodCode}
            onChange={e => setFormData({ ...formData, PodCode: e.target.value })}
            className="glass-input"
          >
            {locations.map(loc => <option key={loc} value={loc}>{loc}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label>Mode</label>
          <select value={formData.ModeOfTransport} onChange={e => setFormData({ ...formData, ModeOfTransport: e.target.value })}>
            <option value="Air">Air</option>
            <option value="Ocean">Ocean</option>
            <option value="Rail">Rail</option>
            <option value="Road">Road</option>
          </select>
        </div>

        <div className="form-group" style={{ marginTop: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><AlertTriangle size={14} color="#f59e0b" /> Port Congestion</label>
            <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>{scenario.congestion}%</span>
          </div>
          <input
            type="range" min="0" max="100"
            value={scenario.congestion}
            onChange={e => setScenario({ ...scenario, congestion: e.target.value })}
            style={{ width: '100%', accentColor: '#f59e0b' }}
          />
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><UploadCloud size={14} color="#3b82f6" /> Weather Severity</label>
            <span style={{ color: '#3b82f6', fontWeight: 'bold' }}>{scenario.weather}%</span>
          </div>
          <input
            type="range" min="0" max="100"
            value={scenario.weather}
            onChange={e => setScenario({ ...scenario, weather: e.target.value })}
            style={{ width: '100%', accentColor: '#3b82f6' }}
          />
        </div>

        {error && <div className="error-banner" style={{ marginTop: '1rem' }}>{error}</div>}

        <div style={{ marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
          <button className="action-btn" onClick={handleSimulate}>
            <Activity size={18} /> Run Simulation
          </button>
        </div>
      </div>

      <div className="glass-card" style={{ minHeight: '400px' }}>
        {!simResult ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', flexDirection: 'column' }}>
            <Activity size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
            <p>Configure trip and run simulation to see Geospatial & AI analysis.</p>
          </div>
        ) : (
          <div className="result-box" style={{ border: 'none', padding: 0, marginTop: 0 }}>
            <div className="result-header">
              <CheckCircle color="#10b981" size={24} />
              <h2>Simulation Complete</h2>
            </div>

            {/* Rich Metrics Grid */}
            {simResult.rich_metrics && (
              <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div className="metric-box">
                  <div className="label">Distance</div>
                  <div className="value">{simResult.rich_metrics.distance}</div>
                </div>
                <div className="metric-box">
                  <div className="label">Avg Speed</div>
                  <div className="value">{simResult.rich_metrics.avg_speed}</div>
                </div>
                <div className="metric-box">
                  <div className="label">Carbon Footprint</div>
                  <div className="value" style={{ color: '#f59e0b' }}>{simResult.rich_metrics.carbon_footprint}</div>
                </div>
              </div>
            )}

            {/* Intelligence Panel (Hackathon Feature) */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
              <div className="metric-box" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6' }}>
                <div className="label" style={{ color: '#93c5fd' }}>Prediction Confidence</div>
                <div className="value" style={{ color: '#fff' }}>{simResult.confidence_score}%</div>
                <div className="progress-bar" style={{ height: '4px', background: '#1e3a8a', marginTop: '5px', borderRadius: '2px' }}>
                  <div style={{ width: `${simResult.confidence_score}%`, height: '100%', background: '#3b82f6' }}></div>
                </div>
              </div>
              <div className="metric-box" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444' }}>
                <div className="label" style={{ color: '#fca5a5' }}>Route Risk Score</div>
                <div className="value" style={{ color: '#fff' }}>{simResult.risk_score} <span style={{ fontSize: '0.8rem', color: '#fca5a5' }}>/ 100</span></div>
                <div className="progress-bar" style={{ height: '4px', background: '#7f1d1d', marginTop: '5px', borderRadius: '2px' }}>
                  <div style={{ width: `${simResult.risk_score}%`, height: '100%', background: '#ef4444' }}></div>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1rem', padding: '1.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
              <div>
                <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Predicted Duration</div>
                <div className="result-time" style={{ fontSize: '3rem', fontWeight: 'bold', color: '#fff' }}>
                  {Math.round(simResult.prediction_hours)} <span style={{ fontSize: '1rem', color: '#64748b', fontWeight: 'normal' }}>hours</span>
                </div>
                <div style={{ color: '#64748b', fontSize: '0.9rem' }}>({simResult.prediction_days} days)</div>
              </div>
              {/* Transshipment Badge */}
              {simResult.route_details?.via_port !== 'DIRECT' && (
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.8rem', color: '#f59e0b', marginBottom: '0.2rem' }}>complex route</div>
                  <div className="badge" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b' }}>
                    {simResult.route_details?.stops_count} STOP(S) VIA {simResult.route_details?.via_port?.replace('|', ' & ')}
                  </div>
                </div>
              )}
            </div>

            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '0.5rem', borderLeft: '4px solid #10b981', textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                <TrendingUp size={18} /> AI Analysis
              </div>
              <div style={{ fontSize: '0.9rem', color: '#e2e8f0', lineHeight: '1.6', whiteSpace: 'pre-line' }}>
                {simResult.ai_explanation}
              </div>
            </div>

            {/* Map Integration */}
            <div style={{ marginTop: '1.5rem', background: '#0f172a', borderRadius: '0.5rem', padding: '0.5rem', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1000, background: 'rgba(0,0,0,0.5)', padding: '0.3rem 0.6rem', borderRadius: '0.3rem', fontSize: '0.7rem', color: '#fff' }}>LIVE TRACKING (OpenStreetMap)</div>
              {simResult.coordinates && simResult.coordinates.source ? (
                <RealWorldMap src={simResult.coordinates.source} dest={simResult.coordinates.destination} mode={formData.ModeOfTransport} />
              ) : (
                <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Map Data Unavailable for this Route</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function UploadView({ API_URL }) {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [learn, setLearn] = useState(false);
  const [status, setStatus] = useState(null);
  const [isSynced, setIsSynced] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setIsSynced(false);
    setPrediction(null);
    setStatus(null);
  };

  const [uploadedFilename, setUploadedFilename] = useState(null);

  const handleSync = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);

    try {
      setStatus('⏳ Uploading to Azure & Checking Quality...');
      const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: fd });
      const data = await res.json();

      if (data.status === 'success') {
        setIsSynced(true);
        setUploadedFilename(data.filename); // Store filename for ingest
        setStatus(`✅ CHECK PASSED (Accuracy: ${data.accuracy}%)\nCloud Upload Complete.\nProceed to Ingestion.`);
      } else if (data.status === 'warning') {
        setIsSynced(false);
        setStatus(`⚠️ CHECK FAILED (Accuracy: ${data.accuracy}%)\nFile Uploaded to Archive Only.\n${data.message}`);
      } else {
        setStatus(`❌ Error: ${data.message}`);
      }
    } catch (err) {
      console.error("Sync Error:", err);
      setStatus(`❌ Sync error: ${err.message}`);
    }
  };

  const handleIngest = async () => {
    if (!uploadedFilename) return;
    try {
      setStatus('⏳ Triggering Ingestion & Transformation...');
      const res = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: uploadedFilename })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setStatus(`✅ ${data.message}\nCheck logs for details.`);
      } else {
        setStatus(`❌ Ingestion failed: ${data.message}`);
      }
    } catch (err) {
      setStatus(`❌ Ingestion error: ${err.message}`);
    }
  };

  const handleProcess = async () => {
    if (!file || !isSynced) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('train', learn);

    try {
      setStatus('Processing file for insights...');
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd });
      const data = await res.json();
      setPrediction(data);
      setStatus('✅ Processing complete. Results displayed below.');
    } catch (err) {
      console.error("Processing Error:", err);
      setStatus(`❌ Processing error: ${err.message}`);
    }
  };

  return (
    <div className="glass-card">
      <div className="upload-section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>

        {/* File Upload Area */}
        <div className="upload-area" style={{ textAlign: 'center', padding: '2rem', border: '2px dashed #94a3b8', borderRadius: '1rem', gridColumn: 'span 3' }}>
          <UploadCloud size={48} color="#3b82f6" />
          <h3 style={{ margin: '1rem 0 0.5rem' }}>Upload Data File</h3>
          <p style={{ color: '#94a3b8', marginBottom: '1.5rem' }}>
            Supported formats: <strong>CSV, Text (.txt), Parquet</strong>
            <br />
            <span style={{ fontSize: '0.9em', color: '#ef4444' }}>(XML and JSON are NOT supported)</span>
          </p>

          <input type="file" onChange={handleFileChange} style={{ display: 'none' }} id="file-upload" accept=".csv,.txt,.parquet" />
          <label htmlFor="file-upload" className="action-btn" style={{ width: '200px', margin: '0 auto', display: 'inline-block' }}>Browse Files</label>

          {file && <p style={{ color: '#10b981', marginTop: '1rem' }}>Selected: {file.name}</p>}

          <div style={{ margin: '1.5rem 0 1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={learn}
              onChange={e => setLearn(e.target.checked)}
              id="learn-toggle"
              style={{ width: 'auto' }}
            />
            <label htmlFor="learn-toggle" style={{ margin: 0, color: '#e2e8f0', cursor: 'pointer' }}>
              Teach model with this data (Continuous Learning)
            </label>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', marginTop: '1rem' }}>
            {/* Step 1: Upload & Check */}
            <button
              className="action-btn"
              onClick={handleSync}
              disabled={!file}
              style={{ background: file ? '#f59e0b' : '#64748b' }}
            >
              1. Upload & Check Quality
            </button>

            {/* Step 2: Ingest (Only if Uploaded/Synced) */}
            {file && isSynced && (
              <button
                className="action-btn"
                onClick={handleIngest}
                style={{ background: '#10b981' }}
              >
                2. Ingest & Transform
              </button>
            )}
          </div>

          {status && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', borderRadius: '0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.95rem', color: '#e2e8f0' }}>
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', margin: 0 }}>{status}</pre>
            </div>
          )}

          {/* Show Process button only if synced/ingested successfully */}
          {file && isSynced && (
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <button className="action-btn" onClick={handleProcess} style={{ background: '#10b981' }}>Proceed to Insights</button>
            </div>
          )}



        </div>

        {/* Database Upload Option */}
        <div className="action-card" style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '1rem', textAlign: 'center', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <Database size={32} color="#f59e0b" />
          </div>
          <h4>Upload from DB Server</h4>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Connect to SQL/NoSQL databases</p>
          <button className="sm-btn" style={{ marginTop: '1rem', width: '100%' }}>Connect</button>
        </div>

        {/* API Fetch Option */}
        <div className="action-card" style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '1rem', textAlign: 'center', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <Webhook size={32} color="#10b981" />
          </div>
          <h4>Fetch Data from API</h4>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Ingest from REST/GraphQL</p>
          <button className="sm-btn" style={{ marginTop: '1rem', width: '100%' }}>Configure</button>
        </div>
      </div>

      {/* Prediction Result Display */}
      {prediction && (
        <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '0.5rem' }}>
          <h4 style={{ margin: '0 0 0.5rem', color: '#10b981' }}>Prediction Result</h4>
          <pre style={{ margin: 0, color: '#e2e8f0' }}>{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function TrackingView({ API_URL }) {
  const [tid, setTid] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeList, setActiveList] = useState([]);

  useEffect(() => {
    fetch(`${API_URL}/active`)
      .then(res => res.json())
      .then(setActiveList)
      .catch(err => console.error("Failed to load active shipments", err));
  }, [API_URL]);

  const handleTrack = async (idOverride) => {
    const targetId = idOverride || tid;
    if (!targetId) return;
    setTid(targetId); // Update input if called via click

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/track/${targetId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Tracking Failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="view-container">
      <div className="card-grid" style={{ gridTemplateColumns: '1fr' }}>
        <div className="glass-card">
          <div className="stat-header">Global Shipment Tracking</div>
          <div className="input-group" style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <input
              type="text"
              placeholder="Enter PNR, Container ID, IMO, or Flight No (e.g. EK 581, TRK-99-AB)"
              value={tid}
              onChange={e => setTid(e.target.value)}
              className="glass-input"
              style={{ flex: 1 }}
            />
            <button className="accent-btn" onClick={() => handleTrack()} disabled={loading}>
              {loading ? "Locating..." : "Track Now"}
            </button>
          </div>
          {error && <div className="error-banner" style={{ marginTop: '1rem' }}>{error}</div>}
        </div>
      </div>

      {result && (
        <div className="glass-card fade-in" style={{ marginTop: '2rem' }}>
          <div className="result-header">
            <CheckCircle color="#10b981" size={24} />
            <h2>Shipment Found: {result.tracking_id}</h2>
          </div>

          <div className="route-visual" style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '2rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', margin: '1rem 0'
          }}>
            <div className="node">
              <div className="node-code">{result.origin.code}</div>
              <div className="node-label">Origin</div>
            </div>
            <div className="connection-line">
              <div className="mode-icon"><Truck size={16} /> {result.mode}</div>
              <div className="line-graphic"></div>
              <div className="metrics-pill">{result.metrics.distance}</div>
            </div>
            <div className="node">
              <div className="node-code">{result.destination.code}</div>
              <div className="node-label">Destination</div>
            </div>
          </div>

          <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
            <div className="metric-box">
              <div className="label">Status</div>
              <div className="value">{result.status}</div>
            </div>
            <div className="metric-box">
              <div className="label">ETA</div>
              <div className="value">{result.eta}</div>
            </div>
            <div className="metric-box">
              <div className="label">Avg Speed</div>
              <div className="value">{result.metrics.avg_speed}</div>
            </div>
            <div className="metric-box">
              <div className="label">Carbon Footprint</div>
              <div className="value" style={{ color: '#f59e0b' }}>{result.metrics.carbon_footprint}</div>
            </div>
          </div>

          {/* Map Integration for Tracking */}
          <div style={{ marginTop: '1.5rem', background: '#0f172a', borderRadius: '0.5rem', padding: '0.5rem', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 1000, background: 'rgba(0,0,0,0.5)', padding: '0.3rem 0.6rem', borderRadius: '0.3rem', fontSize: '0.7rem', color: '#fff' }}>LIVE TRACKING</div>
            {result.coordinates ? (
              <RealWorldMap src={result.coordinates.source} dest={result.coordinates.destination} mode={result.mode} />
            ) : (
              <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Map Data Unavailable</div>
            )}
          </div>
        </div>
      )}

      {/* Active Shipments Table */}
      {!result && activeList.length > 0 && (
        <div className="glass-card" style={{ marginTop: '2rem' }}>
          <div className="stat-header" style={{ marginBottom: '1rem' }}>Live Operations - Top 100 Active</div>
          <div className="table-wrapper" style={{ maxHeight: '400px', overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Origin</th>
                  <th>Target</th>
                  <th>Mode</th>
                  <th>Status (Live)</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {activeList.map((row, idx) => (
                  <tr key={idx} className="hover-row">
                    <td style={{ fontFamily: 'monospace', color: '#3b82f6' }}>{row.id}</td>
                    <td>{row.origin}</td>
                    <td>{row.destination}</td>
                    <td>
                      <span className={`badge ${row.mode.toLowerCase()}`}>{row.mode}</span>
                    </td>
                    <td>{row.status}</td>
                    <td>
                      <button className="sm-btn" onClick={() => handleTrack(row.id)}>Track</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
