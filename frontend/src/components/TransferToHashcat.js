// frontend/src/components/TransferToHashcat.js

import React, {useState} from 'react';
import apiService from '../services/apiService';
import './TransferToHashcat.css'; // Create and style as needed

function TransferToHashcat() {
    const [hashFile, setHashFile] = useState(null);
    const [attackMode, setAttackMode] = useState('straight'); // Options: 'straight', 'combination', 'brute-force'
    const [generatedPasswords, setGeneratedPasswords] = useState('');
    const [hashcatPath, setHashcatPath] = useState('/usr/bin/hashcat'); // Default path, can be made configurable
    const [crackedPasswords, setCrackedPasswords] = useState([]);
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const handleHashFileChange = (event) => {
        setHashFile(event.target.files[0]);
    };

    const handleAttackModeChange = (event) => {
        setAttackMode(event.target.value);
    };

    const handleHashcatPathChange = (event) => {
        setHashcatPath(event.target.value);
    };

    const handleGeneratedPasswordsChange = (event) => {
        setGeneratedPasswords(event.target.value);
    };

    const handleTransfer = async () => {
        if (!hashFile) {
            setMessage('Please select a hash file.');
            return;
        }

        if (!generatedPasswords.trim()) {
            setMessage('Please provide generated passwords.');
            return;
        }

        const generatedPasswordsList = generatedPasswords.split('\n').filter(pwd => pwd.trim() !== '');

        const formData = new FormData();
        formData.append('hash_file', hashFile);
        formData.append('attack_mode', attackMode);
        formData.append('generated_passwords', JSON.stringify(generatedPasswordsList));
        formData.append('hashcat_path', hashcatPath);

        setLoading(true);
        setMessage('');
        setCrackedPasswords([]);

        try {
            const response = await apiService.transferPasswords(hashFile, attackMode, generatedPasswordsList, hashcatPath);
            setCrackedPasswords(response.data.cracked_passwords);
            setMessage('Passwords transferred to Hashcat successfully.');
        } catch (error) {
            setMessage(error.response?.data?.detail || 'An error occurred during transfer.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="transfer-hashcat-container">
            <h2>Transfer Passwords to Hashcat</h2>
            <div className="form-group">
                <label>Hash File:</label>
                <input type="file" accept=".txt,.hash" onChange={handleHashFileChange}/>
            </div>
            <div className="form-group">
                <label>Attack Mode:</label>
                <select value={attackMode} onChange={handleAttackModeChange}>
                    <option value="straight">Straight</option>
                    <option value="combination">Combination</option>
                    <option value="brute-force">Brute-Force</option>
                    {/* Add other attack modes as needed */}
                </select>
            </div>
            <div className="form-group">
                <label>Hashcat Executable Path:</label>
                <input
                    type="text"
                    value={hashcatPath}
                    onChange={handleHashcatPathChange}
                    placeholder="/usr/bin/hashcat"
                />
            </div>
            <div className="form-group">
                <label>Generated Passwords (one per line):</label>
                <textarea
                    rows="10"
                    cols="50"
                    value={generatedPasswords}
                    onChange={handleGeneratedPasswordsChange}
                    placeholder="Enter generated passwords, one per line"
                />
            </div>
            <button onClick={handleTransfer} disabled={loading}>
                {loading ? 'Transferring...' : 'Transfer to Hashcat'}
            </button>
            {message && <p className="message">{message}</p>}
            {crackedPasswords.length > 0 && (
                <div className="cracked-passwords-output">
                    <h3>Cracked Passwords:</h3>
                    <textarea value={crackedPasswords.join('\n')} readOnly rows={10} cols={50}/>
                </div>
            )}
        </div>
    );
}

export default TransferToHashcat;
