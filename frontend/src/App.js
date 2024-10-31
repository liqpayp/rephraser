import React from 'react';
import {BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import TrainModels from './components/TrainModels';
import PasswordGenerator from './components/PasswordGenerator';
import TransferToHashcat from './components/TransferToHashcat';
import Home from './pages/Home';
import About from './pages/About';
import Contact from './pages/Contact';
import HashcatResults from './pages/HashcatResults';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
    return (
        <Router>
            <div className="app-container">
                <Navbar/>
                <div className="content">
                    <Routes>
                        <Route path="/" element={<Home/>}/>
                        <Route path="/train" element={<TrainModels/>}/>
                        <Route path="/generate" element={<PasswordGenerator/>}/>
                        <Route path="/transfer" element={<TransferToHashcat/>}/>
                        <Route path="/about" element={<About/>}/>
                        <Route path="/contact" element={<Contact/>}/>
                        <Route path="/hashcat-results" element={<HashcatResults/>}/>
                    </Routes>
                </div>
                <Footer/>
            </div>
        </Router>
    );
}

export default App;
