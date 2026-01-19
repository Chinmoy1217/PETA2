import { useState } from 'react';
import SignupModal from './SignupModal';

const LoginView = ({ onLogin, API_URL }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showSignup, setShowSignup] = useState(false);

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
        <div style={{ position: 'relative', height: '100vh', width: '100vw', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>

            {/* Background Video */}
            <div style={{
                position: 'absolute',
                top: '-10%', left: '-10%',
                width: '120%', height: '120%',
                zIndex: -1,
                filter: 'brightness(0.5) contrast(1.1)'
            }}>
                <iframe
                    width="100%"
                    height="100%"
                    src="https://www.youtube.com/embed/5ye6OEhLFZc?autoplay=1&mute=1&controls=0&loop=1&playlist=5ye6OEhLFZc&showinfo=0&rel=0&iv_load_policy=3&fs=0&disablekb=1&start=2&end=25"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    style={{ pointerEvents: 'none', objectFit: 'cover' }}
                />
            </div>

            <div className="glass-card" style={{
                width: '400px',
                padding: '2.5rem',
                background: 'rgba(30, 41, 59, 0.7)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
            }}>
                <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
                    <h1 style={{ fontSize: '2rem', fontWeight: '800', margin: 0, textShadow: '0 2px 10px rgba(0,0,0,0.5)' }}>ETA Insight</h1>
                    <p style={{ color: '#94a3b8', margin: '0.5rem 0 0' }}>Next-Gen Logistics Intelligence</p>
                </div>

                <form onSubmit={handleSubmit}>
                    <div style={{ marginBottom: '1.2rem' }}>
                        <label style={{ display: 'block', color: '#cbd5e1', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Username</label>
                        <input
                            type="text"
                            value={username}
                            onChange={e => setUsername(e.target.value)}
                            className="glass-input"
                            style={{ width: '100%', padding: '0.75rem', borderRadius: '0.5rem', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#fff' }}
                            placeholder="admin"
                        />
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', color: '#cbd5e1', marginBottom: '0.5rem', fontSize: '0.9rem' }}>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            className="glass-input"
                            style={{ width: '100%', padding: '0.75rem', borderRadius: '0.5rem', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#fff' }}
                            placeholder="admin"
                        />
                    </div>

                    {error && <div style={{ color: '#ef4444', marginBottom: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', padding: '0.75rem', borderRadius: '0.5rem', fontSize: '0.9rem' }}>{error}</div>}

                    <button type="submit" className="action-btn" style={{ width: '100%', padding: '0.9rem', fontSize: '1rem', marginBottom: '1rem' }}>
                        {loading ? 'Accessing Secure Area...' : 'Login'}
                    </button>
                </form>

                <div style={{ textAlign: 'center', marginTop: '1rem' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                        New user? <button onClick={() => setShowSignup(true)} style={{ background: 'none', border: 'none', color: '#3b82f6', cursor: 'pointer', fontWeight: 'bold' }}>Create Account</button>
                    </p>
                </div>
            </div>

            {showSignup && <SignupModal onClose={() => setShowSignup(false)} API_URL={API_URL} />}
        </div>
    );
};

export default LoginView;
