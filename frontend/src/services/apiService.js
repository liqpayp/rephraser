import axios from 'axios';

const apiClient = axios.create({
    baseURL: 'http://localhost:8000/api', // Hardcode the base URL
    headers: {
        'Content-Type': 'application/json',
    },
});

// Train Models
export const trainModels = (formData) => {
    return apiClient.post('/train/train-models', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
};

// Generate Passwords
export const generatePassword = (model, numPasswords, stream) => {
    return apiClient.post('/generate/generate-password', null, {
        params: {
            model,
            num_passwords: numPasswords,
            stream,
        },
    });
};

// Transfer Passwords to Hashcat
export const transferPasswords = (hashFile, attackMode, generatedPasswords, hashcatPath) => {
    const formData = new FormData();
    formData.append('hash_file', hashFile);
    formData.append('attack_mode', attackMode);
    formData.append('generated_passwords', JSON.stringify(generatedPasswords));
    formData.append('hashcat_path', hashcatPath);

    return apiClient.post('/hashcat/transfer-passwords', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
};

export default {
    trainModels,
    generatePassword,
    transferPasswords,
};
