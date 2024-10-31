// frontend/src/pages/About.js

import React from 'react';
import './About.css'; // Optional: Create a separate CSS file

function About() {
    return (
        <div className="about-page">
            <h2>About HybridModelsProject</h2>
            <p>
                HybridModelsProject is designed to leverage the power of multiple machine learning models to generate
                secure and robust passwords. By integrating models like Markov Chains, RNNs, and GANs, the project
                ensures diversity and strength in password generation.
            </p>
            <p>
                Additionally, the project interfaces with Hashcat to demonstrate password cracking capabilities,
                providing insights into password security and the effectiveness of different hashing algorithms.
            </p>
            <h3>Technologies Used</h3>
            <ul>
                <li>React.js for the frontend</li>
                <li>FastAPI for the backend</li>
                <li>TensorFlow and Keras for machine learning models</li>
                <li>Hashcat for password cracking</li>
                <li>Docker for containerization</li>
            </ul>
        </div>
    );
}

export default About;
