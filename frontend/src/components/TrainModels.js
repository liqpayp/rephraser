// frontend/src/components/TrainModels.js

import React, {useState} from 'react';
import apiService from '../services/apiService';
import './TrainModels.css'; // Create and style as needed

function TrainModels() {
    const [corpusFile, setCorpusFile] = useState(null);
    const [models, setModels] = useState({
        markov: false,
        rnn: false,
        gan: false,
        hybrid: false,
    });
    const [incremental, setIncremental] = useState(false);
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setCorpusFile(event.target.files[0]);
    };

    const handleModelChange = (event) => {
        setModels({
            ...models,
            [event.target.name]: event.target.checked,
        });
    };

    const handleIncrementalChange = (event) => {
        setIncremental(event.target.checked);
    };

    const handleTrain = async () => {
        if (!corpusFile) {
            setMessage('Please select a corpus file.');
            return;
        }

        const selectedModels = Object.keys(models).filter((model) => models[model]);
        if (selectedModels.length === 0) {
            setMessage('Please select at least one model to train.');
            return;
        }

        const formData = new FormData();
        formData.append('corpus_file', corpusFile);
        formData.append('models', JSON.stringify(selectedModels));
        formData.append('incremental', incremental);

        setLoading(true);
        setMessage('');

        try {
            const response = await apiService.trainModels(formData);
            setMessage(response.data.message);
        } catch (error) {
            setMessage(error.response?.data?.detail || 'An error occurred during training.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="train-models-container">
            <h2>Train Models</h2>
            <div className="form-group">
                <label>Corpus File:</label>
                <input type="file" accept=".txt" onChange={handleFileChange}/>
            </div>
            <div className="form-group">
                <label>Select Models to Train:</label>
                <div className="checkbox-group">
                    <label>
                        <input
                            type="checkbox"
                            name="markov"
                            checked={models.markov}
                            onChange={handleModelChange}
                        />
                        Markov Model
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            name="rnn"
                            checked={models.rnn}
                            onChange={handleModelChange}
                        />
                        RNN Model
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            name="gan"
                            checked={models.gan}
                            onChange={handleModelChange}
                        />
                        GAN Model
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            name="hybrid"
                            checked={models.hybrid}
                            onChange={handleModelChange}
                        />
                        Hybrid Model
                    </label>
                </div>
            </div>
            <div className="form-group">
                <label>
                    <input
                        type="checkbox"
                        checked={incremental}
                        onChange={handleIncrementalChange}
                    />
                    Incremental Training
                </label>
            </div>
            <button onClick={handleTrain} disabled={loading}>
                {loading ? 'Training...' : 'Train Models'}
            </button>
            {message && <p className="message">{message}</p>}
        </div>
    );
}

export default TrainModels;
