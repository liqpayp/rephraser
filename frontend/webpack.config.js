const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');
const dotenv = require('dotenv');

// Load environment variables from .env file
const env = dotenv.config().parsed || {};

// Convert the environment variables to a format that DefinePlugin can use
const envKeys = Object.keys(env).reduce((prev, next) => {
    prev[`process.env.${next}`] = JSON.stringify(env[next]);
    return prev;
}, {});

module.exports = {
    entry: './src/index.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.[contenthash].js',
        publicPath: '/', // Important for React Router
    },
    mode: 'development',
    devServer: {
        historyApiFallback: true, // Redirect 404s to index.html
        proxy: {
            '/api': 'http://localhost:8000', // Proxy API requests to backend
        },
        static: path.join(__dirname, 'public'),
        port: 3000,
        open: true,
    },
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/, // Match both .js and .jsx files
                exclude: /node_modules/, // Exclude node_modules
                use: {
                    loader: 'babel-loader',
                    options: {
                        // Optionally, specify presets here
                        presets: ['@babel/preset-env', '@babel/preset-react'],
                    },
                },
            },
            {
                test: /\.css$/, // To handle CSS imports
                use: ['style-loader', 'css-loader'],
            },
            {
                test: /\.(png|svg|jpg|jpeg|gif)$/i, // To handle image imports
                type: 'asset/resource',
            },
            // Add any other loaders you might need
        ],
    },
    resolve: {
        extensions: ['.js', '.jsx'], // Resolve these extensions
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './public/index.html',
        }),
        new webpack.DefinePlugin(envKeys),
    ],
};
