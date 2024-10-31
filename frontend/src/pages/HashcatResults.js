// frontend/src/pages/HashcatResults.js

import React, {useState} from 'react';
import './HashcatResults.css'; // Create and style as needed

function HashcatResults() {
    const [results, setResults] = useState([]);
    const [hashFile, setHashFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    const handleHashFileChange = (event) => {
        setHashFile(event.target.files[0]);
    };

    const handleViewResults = () => {
        if (!hashFile) {
            setMessage('Please select a hash file.');
            return;
        }
        // Implement fetching and displaying results
        // This can involve reading from the backend's hashcat/results directory
        // For security, ensure appropriate endpoints are created in the backend
        // For simplicity, this functionality is left as a placeholder
        setMessage('Feature under development.');
    };

    return (
        <div className="hashcat-results-container">
            <h2>Hashcat Results</h2>
            <div className="form-group">
                <label>Select Hash File to View Results:</label>
                <input type="file" accept=".txt,.hash" onChange={handleHashFileChange}/>
            </div>
            <button onClick={handleViewResults} disabled={loading}>
                {loading ? 'Loading...' : 'View Results'}
            </button>
            {message && <p className="message">{message}</p>}
            {results.length > 0 && (
                <div className="results-output">
                    <h3>Cracked Passwords:</h3>
                    <textarea value={results.join('\n')} readOnly rows={10} cols={50}/>
                </div>
            )}
        </div>
    );
}

export default HashcatResults;
