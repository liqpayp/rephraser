// frontend/src/components/HashcatInterface.js

import React, {useState} from 'react';
import apiService from '../services/apiService';
import './HashcatInterface.css'; // Optional: Create a separate CSS file

function HashcatInterface() {
    const [hashType, setHashType] = useState(0); // Default hash type (e.g., MD5)
    const [hashes, setHashes] = useState('');
    const [wordlist, setWordlist] = useState('');
    const [taskId, setTaskId] = useState('');
    const [status, setStatus] = useState(null);
    const [crackedPasswords, setCrackedPasswords] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);
        setTaskId('');
        setStatus(null);
        setCrackedPasswords([]);

        // Prepare hashes array
        const hashesArray = hashes.split('\n').map(h => h.trim()).filter(h => h !== '');

        try {
            const response = await apiService.submitHashes({
                hash_type: hashType,
                hashes: hashesArray,
                wordlist: wordlist
            });
            setTaskId(response.data.task_id);
            setStatus('Task Started');
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred while submitting hashes.');
        } finally {
            setLoading(false);
        }
    };

    const handleCheckStatus = async () => {
        if (!taskId) return;
        setLoading(true);
        setError(null);
        try {
            const response = await apiService.getHashcatStatus(taskId);
            setStatus(response.data.status);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred while fetching task status.');
        } finally {
            setLoading(false);
        }
    };

    const handleGetResults = async () => {
        if (!taskId) return;
        setLoading(true);
        setError(null);
        try {
            const response = await apiService.getCrackedPasswords(taskId);
            setCrackedPasswords(response.data.cracked_passwords);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred while fetching cracked passwords.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="hashcat-interface">
            <h2>Hashcat Interface</h2>

            <div className="form-group">
                <label htmlFor="hashType">Hash Type:</label>
                <select
                    id="hashType"
                    value={hashType}
                    onChange={(e) => setHashType(Number(e.target.value))}
                >
                    <option value={0}>MD5</option>
                    <option value={100}>SHA1</option>
                    <option value={1400}>SHA256</option>
                    {/* Add more hash types as needed */}
                </select>
            </div>

            <div className="form-group">
                <label htmlFor="hashes">Hashes (one per line):</label>
                <textarea
                    id="hashes"
                    value={hashes}
                    onChange={(e) => setHashes(e.target.value)}
                    rows="5"
                    placeholder="Enter hashes here..."
                ></textarea>
            </div>

            <div className="form-group">
                <label htmlFor="wordlist">Wordlist Path:</label>
                <input
                    type="text"
                    id="wordlist"
                    value={wordlist}
                    onChange={(e) => setWordlist(e.target.value)}
                    placeholder="e.g., wordlists/generated_passwords.txt"
                />
            </div>

            <button onClick={handleSubmit} disabled={loading}>
                {loading ? 'Submitting...' : 'Submit for Cracking'}
            </button>

            {error && <p className="error">{error}</p>}

            {taskId && (
                <div className="task-info">
                    <p><strong>Task ID:</strong> {taskId}</p>
                    <button onClick={handleCheckStatus} disabled={loading}>
                        {loading ? 'Checking...' : 'Check Status'}
                    </button>
                    <button onClick={handleGetResults} disabled={loading || status !== 'completed'}>
                        {loading ? 'Fetching...' : 'Get Results'}
                    </button>
                    {status && <p><strong>Status:</strong> {status}</p>}
                </div>
            )}

            {crackedPasswords.length > 0 && (
                <div className="cracked-passwords">
                    <h3>Cracked Passwords:</h3>
                    <ul>
                        {crackedPasswords.map((pwd, index) => (
                            <li key={index}>{pwd}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default HashcatInterface;
