// frontend/src/pages/Contact.js

import React, {useState} from 'react';
import './Contact.css'; // Optional: Create a separate CSS file

function Contact() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        message: ''
    });
    const [submitted, setSubmitted] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        // For demonstration, we'll just log the data and show a success message
        console.log('Contact Form Data:', formData);
        setSubmitted(true);
        setError(null);
        // Reset form
        setFormData({name: '', email: '', message: ''});
        // In a real application, send the data to the backend or an email service
    };

    return (
        <div className="contact-page">
            <h2>Contact Us</h2>
            {submitted && <p className="success">Your message has been sent successfully!</p>}
            {error && <p className="error">{error}</p>}
            <form onSubmit={handleSubmit} className="contact-form">
                <div className="form-group">
                    <label htmlFor="name">Name:</label>
                    <input
                        type="text"
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        required
                        placeholder="Your Name"
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="email">Email:</label>
                    <input
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        placeholder="you@example.com"
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="message">Message:</label>
                    <textarea
                        id="message"
                        name="message"
                        value={formData.message}
                        onChange={handleChange}
                        required
                        rows="5"
                        placeholder="Your message here..."
                    ></textarea>
                </div>
                <button type="submit">Send Message</button>
            </form>
        </div>
    );
}

export default Contact;
