// Global state
let isTraining = false;
let updateInterval = null;
let charts = {};

// ARC color palette (matches Python COLOR_MAP in arc.py)
const ARC_COLORS = {
    0: '#000000',  // Black
    1: '#0074D9',  // Blue
    2: '#FF4136',  // Red
    3: '#2ECC40',  // Green
    4: '#FFDC00',  // Yellow
    5: '#AAAAAA',  // Gray
    6: '#F012BE',  // Magenta
    7: '#FF851B',  // Orange
    8: '#7FDBFF',  // Light Blue
    9: '#870C25',  // Maroon
};

// Chart configurations
const chartConfig = {
    type: 'line',
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Steps'
                },
                ticks: {
                    maxTicksLimit: 10
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Value'
                }
            }
        }
    }
};

// Initialize charts
function initCharts() {
    // Accuracy Chart (0-100%)
    const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
    charts.accuracy = new Chart(accuracyCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: [],
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.1,
                    borderWidth: 1,
                    pointRadius: 2,
                },
                {
                    label: 'Accuracy (Moving Avg)',
                    data: [],
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            ...chartConfig.options,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Steps'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });

    // Length Chart
    const lengthCtx = document.getElementById('length-chart').getContext('2d');
    charts.length = new Chart(lengthCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Episode Length',
                    data: [],
                    borderColor: 'rgba(16, 185, 129, 0.5)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.1,
                    borderWidth: 1,
                    pointRadius: 2,
                },
                {
                    label: 'Length (Moving Avg)',
                    data: [],
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0,
                }
            ]
        }
    });
}

// Update UI based on training status
function updateStatusUI(status) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');

    if (status) {
        indicator.className = 'status-indicator running';
        statusText.textContent = 'Training Running';
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        indicator.className = 'status-indicator stopped';
        statusText.textContent = 'Not Running';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Format elapsed time
function formatElapsedTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
}

// Fetch and update metrics
async function updateMetrics() {
    try {
        const window = document.getElementById('window-size').value;
        const response = await fetch(`/api/metrics?window=${window}`);
        const data = await response.json();

        // Update charts
        const maxPoints = 100; // Show last 100 points on chart
        const startIdx = Math.max(0, data.steps.length - maxPoints);

        // Accuracy chart
        charts.accuracy.data.labels = data.steps.slice(startIdx);
        charts.accuracy.data.datasets[0].data = data.accuracies.slice(startIdx);
        charts.accuracy.data.datasets[1].data = data.accuracies_ma.slice(startIdx);
        charts.accuracy.update('none');

        // Length chart
        charts.length.data.labels = data.steps.slice(startIdx);
        charts.length.data.datasets[0].data = data.lengths.slice(startIdx);
        charts.length.data.datasets[1].data = data.lengths_ma.slice(startIdx);
        charts.length.update('none');

        // Update latest values
        if (data.latest) {
            document.getElementById('latest-step').textContent = data.latest.step || 0;
            document.getElementById('latest-accuracy').textContent = (data.latest.accuracy || 0).toFixed(1) + '%';
            document.getElementById('latest-accuracy-ma').textContent = (data.latest.accuracy_ma || 0).toFixed(1) + '%';
            document.getElementById('latest-length').textContent = Math.round(data.latest.length || 0);
            document.getElementById('latest-length-ma').textContent = (data.latest.length_ma || 0).toFixed(1);
        }

    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

// Fetch and update console logs
async function updateConsoleLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();

        const consoleOutput = document.getElementById('console-output');
        const autoScroll = document.getElementById('auto-scroll-checkbox').checked;
        
        if (data.logs && data.logs.length > 0) {
            // Clear empty message if present
            if (consoleOutput.querySelector('.console-empty')) {
                consoleOutput.innerHTML = '';
            }

            // Only update if there are new logs
            const currentLineCount = consoleOutput.querySelectorAll('.console-line').length;
            if (data.logs.length !== currentLineCount) {
                // Rebuild the console output
                consoleOutput.innerHTML = data.logs.map(log => 
                    `<div class="console-line">${escapeHtml(log)}</div>`
                ).join('');

                // Auto-scroll to bottom if enabled
                if (autoScroll) {
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                }
            }
        }

    } catch (error) {
        console.error('Error fetching console logs:', error);
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Fetch and update status
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        isTraining = data.is_training;
        updateStatusUI(isTraining);

        document.getElementById('logdir').textContent = data.logdir || '-';
        document.getElementById('num-datapoints').textContent = data.num_datapoints || 0;

        if (data.elapsed_time) {
            document.getElementById('elapsed-time').textContent = formatElapsedTime(data.elapsed_time);
        } else {
            document.getElementById('elapsed-time').textContent = '-';
        }

    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// Start training
async function startTraining() {
    const config = document.getElementById('config-select').value;
    const customArgs = document.getElementById('custom-args').value;

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                config: config,
                custom_args: customArgs
            })
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('Training started successfully');
            isTraining = true;
            updateStatusUI(true);
        } else {
            alert('Error starting training: ' + result.message);
        }
    } catch (error) {
        console.error('Error starting training:', error);
        alert('Error starting training: ' + error.message);
    }
}

// Stop training
async function stopTraining() {
    if (!confirm('Are you sure you want to stop training?')) {
        return;
    }

    try {
        const response = await fetch('/api/stop', {
            method: 'POST',
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('Training stopped successfully');
            isTraining = false;
            updateStatusUI(false);
        } else {
            alert('Error stopping training: ' + result.message);
        }
    } catch (error) {
        console.error('Error stopping training:', error);
        alert('Error stopping training: ' + error.message);
    }
}

// Clear metrics
async function clearMetrics() {
    if (!confirm('Are you sure you want to clear all metrics data?')) {
        return;
    }

    try {
        const response = await fetch('/api/clear', {
            method: 'POST',
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('Metrics cleared successfully');
            // Reset charts
            for (let chart of Object.values(charts)) {
                chart.data.labels = [];
                for (let dataset of chart.data.datasets) {
                    dataset.data = [];
                }
                chart.update();
            }
            
            // Clear console output
            const consoleOutput = document.getElementById('console-output');
            consoleOutput.innerHTML = '<div class="console-empty">No output yet. Start training to see logs...</div>';
            
            // Clear grids
            const gridsContainer = document.getElementById('grids-container');
            gridsContainer.innerHTML = '<div class="grids-empty">Start training to see episode reconstructions...</div>';
        } else {
            alert('Error clearing metrics: ' + result.message);
        }
    } catch (error) {
        console.error('Error clearing metrics:', error);
        alert('Error clearing metrics: ' + error.message);
    }
}

// Render an ARC grid
function renderARCGrid(gridData) {
    if (!gridData || gridData.length === 0) {
        return null;
    }
    
    const height = gridData.length;
    const width = gridData[0].length;
    
    const gridDiv = document.createElement('div');
    gridDiv.className = 'arc-grid';
    gridDiv.style.gridTemplateColumns = `repeat(${width}, 20px)`;
    gridDiv.style.gridTemplateRows = `repeat(${height}, 20px)`;
    
    for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
            const cell = document.createElement('div');
            cell.className = 'arc-cell';
            const colorValue = gridData[row][col];
            cell.style.backgroundColor = ARC_COLORS[colorValue] || '#000000';
            gridDiv.appendChild(cell);
        }
    }
    
    return gridDiv;
}

// Calculate accuracy between two grids
function calculateGridAccuracy(agentGrid, targetGrid) {
    if (!agentGrid || !targetGrid) return 0;
    
    const targetHeight = targetGrid.length;
    const targetWidth = targetGrid[0].length;
    const agentHeight = agentGrid.length;
    const agentWidth = agentGrid[0].length;
    
    let correctCells = 0;
    const totalCells = targetHeight * targetWidth;
    
    // Check overlapping region
    const minHeight = Math.min(agentHeight, targetHeight);
    const minWidth = Math.min(agentWidth, targetWidth);
    
    for (let row = 0; row < minHeight; row++) {
        for (let col = 0; col < minWidth; col++) {
            if (agentGrid[row][col] === targetGrid[row][col]) {
                correctCells++;
            }
        }
    }
    
    // Penalize for wrong size (cells outside overlap are wrong)
    return (correctCells / totalCells) * 100;
}

// Fetch and update grid visualization
async function updateGridVisualization() {
    try {
        const response = await fetch('/api/grids');
        const result = await response.json();
        
        const gridsContainer = document.getElementById('grids-container');
        
        if (result.status === 'no_data') {
            // Keep the empty message
            return;
        }
        
        if (result.status === 'success' && result.data) {
            const data = result.data;
            
            // Clear container
            gridsContainer.innerHTML = '';
            
            // Create sections for each grid
            const testInputSection = document.createElement('div');
            testInputSection.className = 'grid-section';
            testInputSection.innerHTML = '<div class="grid-label">Test Input</div>';
            const testInputGrid = renderARCGrid(data.test_input);
            if (testInputGrid) {
                testInputSection.appendChild(testInputGrid);
                const sizeLabel = document.createElement('div');
                sizeLabel.className = 'grid-sublabel';
                sizeLabel.textContent = `${data.test_input.length}×${data.test_input[0].length}`;
                testInputSection.appendChild(sizeLabel);
            }
            
            const agentOutputSection = document.createElement('div');
            agentOutputSection.className = 'grid-section';
            agentOutputSection.innerHTML = '<div class="grid-label">Agent Output</div>';
            const agentOutputGrid = renderARCGrid(data.agent_output);
            if (agentOutputGrid) {
                agentOutputSection.appendChild(agentOutputGrid);
                const sizeLabel = document.createElement('div');
                sizeLabel.className = 'grid-sublabel';
                sizeLabel.textContent = `${data.agent_output.length}×${data.agent_output[0].length}`;
                agentOutputSection.appendChild(sizeLabel);
            }
            
            const targetOutputSection = document.createElement('div');
            targetOutputSection.className = 'grid-section';
            targetOutputSection.innerHTML = '<div class="grid-label">Target Output</div>';
            const targetOutputGrid = renderARCGrid(data.test_output);
            if (targetOutputGrid) {
                targetOutputSection.appendChild(targetOutputGrid);
                const sizeLabel = document.createElement('div');
                sizeLabel.className = 'grid-sublabel';
                sizeLabel.textContent = `${data.test_output.length}×${data.test_output[0].length}`;
                targetOutputSection.appendChild(sizeLabel);
            }
            
            gridsContainer.appendChild(testInputSection);
            gridsContainer.appendChild(agentOutputSection);
            gridsContainer.appendChild(targetOutputSection);
            
            // Add stats
            const accuracy = calculateGridAccuracy(data.agent_output, data.test_output);
            const statsDiv = document.createElement('div');
            statsDiv.className = 'grid-stats';
            statsDiv.innerHTML = `
                <div class="grid-stat">
                    <span class="grid-stat-value">${accuracy.toFixed(1)}%</span>
                    <span>Accuracy</span>
                </div>
                <div class="grid-stat">
                    <span class="grid-stat-value">${data.final_reward.toFixed(3)}</span>
                    <span>Reward</span>
                </div>
                <div class="grid-stat">
                    <span class="grid-stat-value">${data.steps}</span>
                    <span>Steps</span>
                </div>
            `;
            
            // Add stats as a full-width section
            const statsSection = document.createElement('div');
            statsSection.style.width = '100%';
            statsSection.style.display = 'flex';
            statsSection.style.justifyContent = 'center';
            statsSection.appendChild(statsDiv);
            gridsContainer.appendChild(statsSection);
        }
    } catch (error) {
        console.error('Error fetching grid visualization:', error);
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initCharts();

    // Set up button handlers
    document.getElementById('start-btn').addEventListener('click', startTraining);
    document.getElementById('stop-btn').addEventListener('click', stopTraining);
    document.getElementById('clear-btn').addEventListener('click', clearMetrics);

    // Update window size when changed
    document.getElementById('window-size').addEventListener('change', updateMetrics);

    // Initial status update
    updateStatus();
    updateMetrics();
    updateConsoleLogs();
    updateGridVisualization();

    // Set up periodic updates
    updateInterval = setInterval(() => {
        updateStatus();
        updateConsoleLogs();  // Always update logs when training
        updateGridVisualization();  // Always update grids
        if (isTraining) {
            updateMetrics();
        }
    }, 2000); // Update every 2 seconds
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});

