import { useState, useEffect } from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, LineChart, Line, AreaChart, Area, PieChart, Pie
} from 'recharts';
import { LayoutDashboard, UploadCloud, Calculator, Activity, Ship, CheckCircle, TrendingUp, AlertTriangle, Download, Menu, X, MessageSquare, Bell } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';
import cozentusLogo from './assets/cozentus_logo.png';

// Fix Leaflet Icon Issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [metrics, setMetrics] = useState(null);
  const [plots, setPlots] = useState(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const API_URL = "http://127.0.0.1:8000";

  useEffect(() => {
    fetch(`${API_URL}/metrics`).then(res => res.json()).then(setMetrics);
    fetch(`${API_URL}/plots`).then(res => res.json()).then(setPlots);
  }, []);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setIsMenuOpen(false); // Close drawer on selection
  };

  // Tab Title Logic
  const getTitle = () => {
    switch (activeTab) {
      case 'summary': return 'Platform Overview';
      case 'dashboard': return 'Operations Overview';
      case 'tracking': return 'Real-Time Tracking';
      case 'simulator': return 'Instant ETA Calculator';
      case 'upload': return 'Batch Processing';
      default: return 'ETA Insight';
    }
  };

  return (
    <div className="app-container">
      {/* Top Menu Bar */}
      <div className="top-bar">
        <button className="menu-btn" onClick={() => setIsMenuOpen(true)}>
          <Menu size={24} color="#0f172a" />
        </button>
        <div className="brand">
          <Ship className="brand-icon" size={24} />
          <span>ETA Insight</span>
        </div>
        <div style={{ flex: 1 }}></div>
        <div style={{ flex: 1 }}></div>

        {/* Action Icons */}
        <div style={{ display: 'flex', gap: '1rem', marginRight: '1.5rem' }}>
          <button className="icon-btn-header" style={{ background: 'none', border: 'none', cursor: 'pointer', position: 'relative' }}>
            <MessageSquare size={20} color="#64748b" />
          </button>
          <button className="icon-btn-header" style={{ background: 'none', border: 'none', cursor: 'pointer', position: 'relative' }}>
            <Bell size={20} color="#64748b" />
            <span style={{ position: 'absolute', top: -2, right: -2, width: '8px', height: '8px', background: '#ef4444', borderRadius: '50%' }}></span>
          </button>
        </div>

        <img src={cozentusLogo} alt="Cozentus Logo" className="header-logo" />
      </div>

      {/* Slide-in Drawer */}
      <div className={`drawer-overlay ${isMenuOpen ? 'open' : ''}`} onClick={() => setIsMenuOpen(false)}></div>
      <nav className={`side-drawer ${isMenuOpen ? 'open' : ''}`}>
        <div className="drawer-header">
          <h2>ETA Insight</h2>
          <button className="close-btn" onClick={() => setIsMenuOpen(false)}>
            <X size={24} />
          </button>
        </div>
        <ul className="nav-links">
          <li className={`nav-item ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => handleTabChange('summary')}>
            <Activity size={20} /> Key Figures
          </li>
          <li className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => handleTabChange('dashboard')}>
            <LayoutDashboard size={20} /> Dashboard
          </li>
          <li className={`nav-item ${activeTab === 'tracking' ? 'active' : ''}`} onClick={() => handleTabChange('tracking')}>
            <CheckCircle size={20} /> Track Shipment
          </li>
          <li className={`nav-item ${activeTab === 'simulator' ? 'active' : ''}`} onClick={() => handleTabChange('simulator')}>
            <Calculator size={20} /> Trip Simulator
          </li>
          <li className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => handleTabChange('upload')}>
            <UploadCloud size={20} /> Batch Predict
          </li>
        </ul>
      </nav>

      {/* Main Content Area */}
      <main className="main-content with-top-bar">
        {/* Fade Transition Wrapper */}
        <div key={activeTab} className="fade-in-content">

          {activeTab === 'summary' && <SummaryView metrics={metrics} />}
          {activeTab === 'dashboard' && <DashboardView metrics={metrics} plots={plots} />}
          {activeTab === 'tracking' && <TrackingView API_URL={API_URL} />}
          {activeTab === 'simulator' && <SimulatorView API_URL={API_URL} />}
          {activeTab === 'upload' && <UploadView API_URL={API_URL} />}
        </div>
      </main>
    </div>
  );
}

function SummaryView({ metrics }) {
  if (!metrics) return <div style={{ padding: '2rem' }}>Loading Data...</div>;

  // Format huge numbers
  const fmt = (n) => {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n;
  };

  return (
    <div className="summary-container" style={{
      background: '#ffffff',
      borderRadius: '20px',
      padding: '1.5rem 3rem',
      textAlign: 'center',
      color: '#1e293b',
      minHeight: '600px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      {/* Cozentus Animation */}
      <div style={{ marginBottom: '1rem' }}>
        <img
          src="https://www.cozentus.com/uploads/images/sequence-031.gif"
          alt="Cozentus Animation"
          style={{
            maxWidth: '500px',
            width: '100%',
            height: 'auto'
          }}
        />
      </div>

      <div style={{
        background: 'rgba(255, 255, 255, 0.25)',
        backdropFilter: 'blur(4px)',
        padding: '0.5rem 1.5rem',
        borderRadius: '999px',
        display: 'inline-block',
        fontSize: '0.9rem',
        fontWeight: 'bold',
        letterSpacing: '1px',
        marginBottom: '0.75rem',
        color: '#475569'
      }}>
        KEY FIGURES
      </div>

      <h1 style={{ fontSize: '2rem', fontWeight: '800', marginBottom: '2rem', color: '#0f172a' }}>
        Data at the core of our platform
      </h1>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '1.5rem',
        width: '100%'
      }}>
        {/* Metric 1 */}
        <div style={{ padding: '0 1rem' }}>
          <div style={{ fontSize: '4rem', fontWeight: '800', marginBottom: '1rem', color: '#0f172a' }}>
            +{fmt(metrics.total_shipments || 0)}
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>
            transport data
          </div>
          <p style={{ color: '#475569', lineHeight: '1.6', fontSize: '0.95rem' }}>
            Derived from real, anonymized, continuously updated flows for optimal representativeness
          </p>
        </div>

        {/* Metric 2 */}
        <div style={{ padding: '0 1rem' }}>
          <div style={{ fontSize: '4rem', fontWeight: '800', marginBottom: '1rem', color: '#0f172a' }}>
            +{metrics.connected_carriers_count || 50}
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>
            connected carriers
          </div>
          <p style={{ color: '#475569', lineHeight: '1.6', fontSize: '0.95rem' }}>
            A network of major global carriers and regional shippers ready to meet your needs
          </p>
        </div>

        {/* Metric 3 */}
        <div style={{ padding: '0 1rem' }}>
          <div style={{ fontSize: '4rem', fontWeight: '800', marginBottom: '1rem', color: '#0f172a' }}>
            +{metrics.delayed_rate || 0}%
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>
            average deviation detected
          </div>
          <p style={{ color: '#475569', lineHeight: '1.6', fontSize: '0.95rem' }}>
            Identify optimization opportunities and regain control over your costs and lead times
          </p>
        </div>
      </div>

      {/* Explanation Section */}
      <div style={{
        marginTop: '2rem',
        textAlign: 'left',
        width: '100%',
        animation: 'fadeInUp 0.6s ease-out 0.3s backwards'
      }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: '700', color: '#1e293b', marginBottom: '1rem', borderBottom: '2px solid rgba(0,0,0,0.1)', paddingBottom: '0.5rem' }}>
          How Insights Are Calculated
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#334155', marginBottom: '0.5rem' }}>
              <TrendingUp size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
              Historical Analysis
            </h4>
            <p style={{ color: '#64748b', fontSize: '0.9rem', lineHeight: '1.6' }}>
              We analyze over {metrics.total_shipments ? (metrics.total_shipments / 1000).toFixed(0) + 'K' : '600K'} historical shipment records to establish baseline performance patterns for every route, carrier, and mode of transport.
            </p>
          </div>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#334155', marginBottom: '0.5rem' }}>
              <Calculator size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
              Live Deviation Tracking
            </h4>
            <p style={{ color: '#64748b', fontSize: '0.9rem', lineHeight: '1.6' }}>
              Our XGBoost v1.0 Model continuously compares live ETA updates against our predictive baseline. The "Average Deviation" KPIs reflect the real-time gap between carrier promises and actual performance.
            </p>
          </div>
        </div>
      </div>
    </div >
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
      {/* 1. CORE KPIs */}
      <h3 style={{ color: '#0f172a', fontWeight: 'bold', borderBottom: '1px solid #334155', paddingBottom: '0.3rem', marginBottom: '0.8rem', fontSize: '0.9rem' }}>Core Performance</h3>
      <div className="card-grid" style={{ marginBottom: '1rem', gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>📦</span> Total Shipments</div>
          <div className="stat-value">{metrics.total_shipments?.toLocaleString()}</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>✅</span> On-Time Delivery %</div>
          <div className="stat-value" style={{ color: '#10b981' }}>{metrics.on_time_rate}%</div>
          <div className="sub-text">ETA Accuracy</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>🕒</span> Late Shipments</div>
          <div className="stat-value" style={{ color: '#ef4444' }}>{metrics.late_shipments_count?.toLocaleString()}</div>
          <div className="sub-text">{metrics.delayed_rate}% of Total</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>⏳</span> Avg Delay</div>
          <div className="stat-value" style={{ color: '#f59e0b' }}>{metrics.avg_delay_days} days</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>🚨</span> Max Delay</div>
          <div className="stat-value" style={{ color: '#ef4444' }}>{metrics.max_delay_days} days</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>⚠️</span> Critical Delays</div>
          <div className="stat-value" style={{ color: '#b91c1c' }}>{metrics.critical_delays_count}</div>
          <div className="sub-text">&gt; 3 Days</div>
        </div>
      </div>

      {/* 2. VARIANCE KPIs */}
      <h3 style={{ color: '#0f172a', fontWeight: 'bold', borderBottom: '1px solid #334155', paddingBottom: '0.3rem', marginBottom: '0.8rem', marginTop: '0.8rem', fontSize: '0.9rem' }}>ETA Variance</h3>
      <div className="card-grid" style={{ marginBottom: '1rem', gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>🔄</span> Avg Variance</div>
          <div className="stat-value">{metrics.avg_eta_variance_days} days</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>🚀</span> Early Arrivals</div>
          <div className="stat-value" style={{ color: '#3b82f6' }}>{metrics.early_arrival_rate}%</div>
        </div>
        <div className="glass-card">
          <div className="stat-header"><span style={{ fontSize: '1.4rem', marginRight: '8px' }}>🎯</span> On-Time Arrivals</div>
          <div className="stat-value" style={{ color: '#10b981' }}>{metrics.on_time_arrival_rate}%</div>
        </div>
      </div>

      {/* 3. OPERATIONAL KPIs (Charts) */}
      <h3 style={{ color: '#0f172a', fontWeight: 'bold', borderBottom: '1px solid #334155', paddingBottom: '0.3rem', marginTop: '0.8rem', marginBottom: '0.8rem', fontSize: '0.9rem' }}>Operational Accuracy</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.8rem', marginBottom: '1rem' }}>

        {/* Mode Accuracy */}
        {/* Mode Accuracy - Donut Chart */}
        <div className="glass-card chart-container">
          <div className="stat-header">By Transport Mode (Donut)</div>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie
                data={metrics.mode_accuracy || []}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {(metrics.mode_accuracy || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={['#3b82f6', '#10b981', '#f59e0b', '#ef4444'][index % 4]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Carrier Accuracy */}
        <div className="glass-card chart-container">
          <div className="stat-header">By Carrier</div>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie
                data={metrics.carrier_accuracy || []}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={50}
                paddingAngle={5}
                dataKey="value"
              >
                {(metrics.carrier_accuracy || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][index % 5]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Route Accuracy */}
        <div className="glass-card chart-container">
          <div className="stat-header">By Route</div>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={metrics.route_accuracy || []} layout="vertical">
              <XAxis type="number" domain={[80, 100]} hide />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" width={80} style={{ fontSize: '0.8rem' }} />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Bar dataKey="value" fill="#f59e0b" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Legacy Charts & Scatter */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', marginBottom: '1rem' }}>
        <div className="glass-card chart-container">
          <div className="stat-header">Model Battle (Accuracy %)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={comparison} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
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
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={plots?.mode_performance || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="mode" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[60, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
        <div className="glass-card chart-container">
          <div className="stat-header">Variance Map (Predicted vs Actual)</div>
          {!plots?.scatter_data ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Loading Data...</div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <ScatterChart key={plots.scatter_data.length}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" dataKey="Actual" name="Actual" stroke="#94a3b8" unit="h" />
                <YAxis type="number" dataKey="Pred" name="Predicted" stroke="#94a3b8" unit="h" />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#3b82f6', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
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
            <ResponsiveContainer width="100%" height={180}>
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
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#10b981', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
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
    <div style={{ height: '100%', width: '100%', borderRadius: '0.5rem', overflow: 'hidden' }}>
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
    <div className="simulator-container" style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2fr', gap: '2rem', alignItems: 'start' }}>
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
            style={{ width: '100%', accentColor: '#f59e0b', padding: 0 }}
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
            style={{ width: '100%', accentColor: '#3b82f6', padding: 0 }}
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
                <div className="label">Prediction Confidence</div>
                <div className="value">{simResult.confidence_score}%</div>
                <div className="progress-bar" style={{ height: '4px', background: '#1e3a8a', marginTop: '5px', borderRadius: '2px' }}>
                  <div style={{ width: `${simResult.confidence_score}%`, height: '100%', background: '#3b82f6' }}></div>
                </div>
              </div>
              <div className="metric-box" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444' }}>
                <div className="label">Route Risk Score</div>
                <div className="value">{simResult.risk_score} <span style={{ fontSize: '0.8rem' }}>/ 100</span></div>
                <div className="progress-bar" style={{ height: '4px', background: '#7f1d1d', marginTop: '5px', borderRadius: '2px' }}>
                  <div style={{ width: `${simResult.risk_score}%`, height: '100%', background: '#ef4444' }}></div>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1rem', padding: '1.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
              <div>
                <div style={{ color: '#aaa', fontSize: '0.9rem' }}>Predicted Duration</div>
                <div className="result-time" style={{ fontSize: '3rem', fontWeight: 'bold' }}>
                  {Math.round(simResult.prediction_hours)} <span style={{ fontSize: '1rem', fontWeight: 'normal' }}>hours</span>
                </div>
                <div className="duration-text" style={{ fontSize: '0.9rem' }}>({simResult.prediction_days} days)</div>
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
              <div className="ai-analysis-text" style={{ fontSize: '0.9rem', lineHeight: '1.6', whiteSpace: 'pre-line' }}>
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

  const handleUpload = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('train', learn); // Send training flag
    const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd });
    setPrediction(await res.json());
  };

  return (
    <div className="glass-card batch-container">
      <div className="upload-area" style={{ textAlign: 'center', padding: '2rem', border: '2px dashed #94a3b8', borderRadius: '1rem', marginBottom: '2rem' }}>
        <UploadCloud size={48} color="#3b82f6" />
        <p>Drag and drop CSV here or click to browse</p>
        <input type="file" onChange={e => setFile(e.target.files[0])} style={{ display: 'none' }} id="file-upload" />
        <label htmlFor="file-upload" className="action-btn" style={{ width: '200px', margin: '1rem auto' }}>Browse Files</label>
        {file && <p style={{ color: '#10b981' }}>Selected: {file.name}</p>}

        <div style={{ margin: '1rem 0', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
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

        {file && <button className="action-btn" onClick={handleUpload}>Process File</button>}
      </div>

      {prediction && (
        <>
          {prediction.ai_summary && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '0.5rem', borderLeft: '4px solid #10b981' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981', fontWeight: 'bold', marginBottom: '0.3rem' }}>
                <TrendingUp size={18} /> Process Analysis
              </div>
              <p style={{ margin: '0.5rem 0 0 0', color: '#e2e8f0', lineHeight: 1.5 }}>{prediction.ai_summary}</p>
            </div>
          )}
          <h3>Analysis Results ({prediction.file_accuracy}% Accuracy on this file)</h3>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Predicted (Hrs)</th>
                  <th>Actual (Hrs)</th>
                  <th>Diff</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {prediction.predictions.map((row) => (
                  <tr key={row.id} className={row.error > 5 ? 'alert-row' : ''}>
                    <td>#{row.id}</td>
                    <td>{row.prediction}</td>
                    <td>{row.actual || '-'}</td>
                    <td>{row.error || '-'}</td>
                    <td>
                      {row.error > 5 ?
                        <span style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><AlertTriangle size={14} /> Deviation</span> :
                        <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><CheckCircle size={14} /> On Track</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function TrackingView({ API_URL }) {
  const [selectedPol, setSelectedPol] = useState('USLAX');
  const [selectedPod, setSelectedPod] = useState('CNSHA');
  const [locations, setLocations] = useState([]);
  const [activeList, setActiveList] = useState([]);
  const [polCoords, setPolCoords] = useState(null);
  const [podCoords, setPodCoords] = useState(null);

  // Load locations
  useEffect(() => {
    fetch(`${API_URL}/locations`)
      .then(res => res.json())
      .then(data => {
        setLocations(data);
        if (data.length > 0) {
          setSelectedPol(data.includes('USLAX') ? 'USLAX' : data[0]);
          setSelectedPod(data.includes('CNSHA') ? 'CNSHA' : (data[1] || data[0]));
        }
      })
      .catch(err => console.error("Failed to load locations", err));
  }, [API_URL]);

  // Load active shipments
  useEffect(() => {
    fetch(`${API_URL}/active`)
      .then(res => res.json())
      .then(setActiveList)
      .catch(err => console.error("Failed to load active shipments", err));
  }, [API_URL]);

  // Fetch coordinates when POL/POD changes
  useEffect(() => {
    if (!selectedPol || !selectedPod) return;

    // Make a prediction call to get coordinates
    fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        pol: selectedPol,
        pod: selectedPod,
        mode: 'Ocean',
        train: 'false'
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.predictions?.[0]?.coordinates) {
          setPolCoords(data.predictions[0].coordinates.source);
          setPodCoords(data.predictions[0].coordinates.destination);
        }
      })
      .catch(err => console.error("Failed to fetch coordinates", err));
  }, [selectedPol, selectedPod, API_URL]);

  // Filter active shipments by POL/POD
  const filteredShipments = activeList.filter(shipment => {
    if (!selectedPol && !selectedPod) return true;
    const matchesPol = !selectedPol || shipment.origin === selectedPol;
    const matchesPod = !selectedPod || shipment.destination === selectedPod;
    return matchesPol && matchesPod;
  });

  return (
    <div className="view-container">
      {/* Filter Controls */}
      <div className="glass-card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label style={{ color: '#1e293b', fontWeight: '600' }}>Origin Port (POL)</label>
            <select
              value={selectedPol}
              onChange={e => setSelectedPol(e.target.value)}
              style={{
                width: '100%',
                padding: '0.8rem',
                background: '#fff',
                border: '1px solid #cbd5e1',
                borderRadius: '0.5rem',
                color: '#0f172a',
                fontSize: '1rem'
              }}
            >
              <option value="">All Origins</option>
              {locations.map(loc => <option key={loc} value={loc}>{loc}</option>)}
            </select>
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label style={{ color: '#1e293b', fontWeight: '600' }}>Destination Port (POD)</label>
            <select
              value={selectedPod}
              onChange={e => setSelectedPod(e.target.value)}
              style={{
                width: '100%',
                padding: '0.8rem',
                background: '#fff',
                border: '1px solid #cbd5e1',
                borderRadius: '0.5rem',
                color: '#0f172a',
                fontSize: '1rem'
              }}
            >
              <option value="">All Destinations</option>
              {locations.map(loc => <option key={loc} value={loc}>{loc}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Map Display */}
      <div className="glass-card" style={{ marginBottom: '2rem' }}>
        <div className="stat-header" style={{ marginBottom: '1rem', color: '#1e293b', fontWeight: 'bold' }}>
          Route Visualization
        </div>
        <div style={{ background: '#f8fafc', borderRadius: '0.5rem', padding: '0.5rem', position: 'relative', overflow: 'hidden', border: '1px solid #cbd5e1', height: 'calc(100vh - 280px)' }}>
          {polCoords && podCoords ? (
            <RealWorldMap src={polCoords} dest={podCoords} mode="Ocean" />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
              Select POL and POD to view route
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
