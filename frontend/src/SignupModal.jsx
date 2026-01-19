import { useState } from 'react';

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
            <div className="glass-card" style={{ width: '350px', padding: '2rem', background: '#1e293b', border: '1px solid #3b82f6', color: '#fff' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h2 style={{ margin: 0, color: '#fff' }}>Sign Up</h2>
                    <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '1.5rem' }}>&times;</button>
                </div>

                {msg ? <div style={{ color: '#10b981', textAlign: 'center', padding: '2rem 0' }}>{msg}</div> : (
                    <form onSubmit={handleSignup}>
                        <div style={{ marginBottom: '1rem' }}>
                            <input type="text" placeholder="Choose Username" value={username} onChange={e => setUsername(e.target.value)} className="glass-input" style={{ width: '100%', boxSizing: 'border-box', color: '#fff' }} required />
                        </div>
                        <div style={{ marginBottom: '1rem' }}>
                            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="glass-input" style={{ width: '100%', boxSizing: 'border-box', color: '#fff' }} required />
                        </div>
                        <div style={{ marginBottom: '1.5rem' }}>
                            <input type="password" placeholder="Confirm Password" value={confirm} onChange={e => setConfirm(e.target.value)} className="glass-input" style={{ width: '100%', boxSizing: 'border-box', color: '#fff' }} required />
                        </div>
                        {error && <div style={{ color: '#ef4444', marginBottom: '1rem', fontSize: '0.9rem' }}>{error}</div>}
                        <button type="submit" className="action-btn" style={{ width: '100%' }}>Register</button>
                    </form>
                )}
            </div>
        </div>
    );
}

export default SignupModal;
