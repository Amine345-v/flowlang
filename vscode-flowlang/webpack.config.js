/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

//@ts-check
'use strict';

const path = require('path');

const withDefaults = (/** @type {any} */ target) => ({
    mode: 'production', // minimal size
    devtool: 'source-map',
    resolve: {
        extensions: ['.ts', '.js'],
        mainFields: ['module', 'main'], // prefer module if available
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: 'ts-loader',
                        options: {
                            transpileOnly: true // faster
                        }
                    }
                ]
            }
        ]
    },
    output: {
        filename: '[name].js',
        path: path.join(__dirname, target, 'dist'),
        libraryTarget: 'commonjs'
    },
    target: 'node',
    externals: {
        'vscode': 'commonjs vscode', // ignored because it's available in the runtime
        'utf-8-validate': 'commonjs utf-8-validate', // optional dep of ws
        'bufferutil': 'commonjs bufferutil' // optional dep of ws
    },
    optimization: {
        minimize: true
    }
});

const clientConfig = {
    context: path.join(__dirname, 'client'),
    entry: {
        extension: './src/extension.ts'
    },
    ...withDefaults('client')
};

const serverConfig = {
    context: path.join(__dirname, 'server'),
    entry: {
        server: './src/server.ts'
    },
    ...withDefaults('server')
};

module.exports = [clientConfig, serverConfig];
