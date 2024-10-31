// frontend/src/components/Navbar.js

import React from 'react';
import {Link} from 'react-router-dom';
import './Navbar.css'; // Create and style as needed

function Navbar() {
    return (
        <nav>
            <ul>
                <li>
                    <Link to="/">Home</Link>
                </li>
                <li>
                    <Link to="/train">Train Models</Link>
                </li>
                <li>
                    <Link to="/generate">Generate Passwords</Link>
                </li>
                <li>
                    <Link to="/transfer">Transfer to Hashcat</Link>
                </li>
                <li>
                    <Link to="/hashcat-results">Hashcat Results</Link>
                </li>
                <li>
                    <Link to="/about">About</Link>
                </li>
                <li>
                    <Link to="/contact">Contact</Link>
                </li>
            </ul>
        </nav>
    );
}

export default Navbar;
