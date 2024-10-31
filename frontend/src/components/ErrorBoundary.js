import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = {hasError: false, error: null};
    }

    static getDerivedStateFromError(error) {
        // Update state to display fallback UI
        return {hasError: true, error};
    }

    componentDidCatch(error, errorInfo) {
        // Log error details
        console.error("ErrorBoundary caught an error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return <h2>Something went wrong: {this.state.error.message}</h2>;
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
