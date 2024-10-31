// frontend/src/components/PasswordGenerator.js

import React, {useState} from 'react';
import apiService from '../services/apiService';
import './PasswordGenerator.css'; // Create and style as needed

function PasswordGenerator() {
    const [selectedModel, setSelectedModel] = useState('markov');
    const [numPasswords, setNumPasswords] = useState(1);
    const [stream, setStream] = useState(false);
    const [passwords, setPasswords] = useState([]);
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    const handleNumPasswordsChange = (event) => {
        setNumPasswords(parseInt(event.target.value));
    };

    const handleStreamChange = (event) => {
        setStream(event.target.checked);
    };

    const handleGenerate = async () => {
        if (numPasswords < 1) {
            setMessage('Number of passwords must be at least 1.');
            return;
        }

        setLoading(true);
        setMessage('');
        setPasswords([]);

        try {
            const response = await apiService.generatePassword(selectedModel, numPasswords, stream);
            setPasswords(response.data.passwords);
            setMessage('Passwords generated successfully.');
        } catch (error) {
            setMessage(error.response?.data?.detail || 'An error occurred during password generation.');
        } finally {
            setLoading(false);
        }
    };

    const handleDownload = () => {
        const element = document.createElement('a');
        const file = new Blob([passwords.join('\n')], {type: 'text/plain'});
        element.href = URL.createObjectURL(file);
        element.download = 'generated_passwords.txt';
        document.body.appendChild(element);
        element.click();
    };

    return (
        <div className="password-generator-container">
            <h2>Generate Passwords</h2>
            <div className="form-group">
                <label>Select Model:</label>
                <select value={selectedModel} onChange={handleModelChange}>
                    <option value="markov">Markov Model</option>
                    <option value="rnn">RNN Model</option>
                    <option value="gan">GAN Model</option>
                    <option value="hybrid">Hybrid Model</option>
                </select>
            </div>
            <div className="form-group">
                <label>Number of Passwords:</label>
                <input
                    type="number"
                    min="1"
                    value={numPasswords}
                    onChange={handleNumPasswordsChange}
                />
            </div>
            <div className="form-group">
                <label>
                    <input
                        type="checkbox"
                        checked={stream}
                        onChange={handleStreamChange}
                    />
                    Streaming Generation Mode
                </label>
            </div>
            <button onClick={handleGenerate} disabled={loading}>
                {loading ? 'Generating...' : 'Generate'}
            </button>
            {message && <p className="message">{message}</p>}
            {passwords.length > 0 && (
                <div className="passwords-output">
                    <h3>Generated Passwords:</h3>
                    <textarea value={passwords.join('\n')} readOnly rows={10} cols={50}/>
                    <br/>
                    <button onClick={handleDownload}>Download Passwords</button>
                </div>
            )}
        </div>
    );
}

export default PasswordGenerator;
