// frontend/src/pages/Home.js

import React from 'react';
import {Link} from 'react-router-dom';
import './Home.css'; // Optional: Create a separate CSS file

function Home() {
    return (
        <div className="home-page">
            <h1>Welcome to HybridModelsProject</h1>
            <p>
                Our project integrates advanced machine learning models to generate secure passwords and interact with
                Hashcat for password cracking tasks.
            </p>
            <div className="home-buttons">
                <Link to="/password-generator">
                    <button>Generate Password</button>
                </Link>
                <Link to="/hashcat-interface">
                    <button>Hashcat Interface</button>
                </Link>
            </div>
        </div>
    );
}

export default Home;
