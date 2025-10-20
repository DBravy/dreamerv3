// Global state
let isTraining = false;
let updateInterval = null;
let charts = {};

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

// Fetch and update grid visualizations
async function updateGridVisualizations() {
    try {
        const limit = document.getElementById('grid-limit').value;
        const response = await fetch(`/api/grids?limit=${limit}`);
        const data = await response.json();

        const gridsGallery = document.getElementById('grids-gallery');
        
        if (data.grids && data.grids.length > 0) {
            // Clear empty message if present
            if (gridsGallery.querySelector('.grids-empty')) {
                gridsGallery.innerHTML = '';
            }

            // Rebuild the gallery
            gridsGallery.innerHTML = data.grids.map(grid => `
                <div class="grid-item">
                    <div class="grid-image-container">
                        <img src="/static/grids/${grid.filename}" alt="Grid at step ${grid.step}" />
                        <div class="grid-labels">
                            <span class="grid-label-left">Test Input (${grid.input_shape[0]}×${grid.input_shape[1]})</span>
                            <span class="grid-label-right">Agent Output (${grid.output_shape[0]}×${grid.output_shape[1]})</span>
                        </div>
                    </div>
                    <div class="grid-info">
                        <div class="grid-info-item">
                            <span class="grid-info-label">Step:</span>
                            <span class="grid-info-value">${grid.step}</span>
                        </div>
                        <div class="grid-info-item">
                            <span class="grid-info-label">Accuracy:</span>
                            <span class="grid-info-value">${(grid.accuracy * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `).join('');
        } else if (!gridsGallery.querySelector('.grids-empty')) {
            gridsGallery.innerHTML = '<div class="grids-empty">No grids yet. Start training to see reconstructions...</div>';
        }

    } catch (error) {
        console.error('Error fetching grid visualizations:', error);
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
        } else {
            alert('Error clearing metrics: ' + result.message);
        }
    } catch (error) {
        console.error('Error clearing metrics:', error);
        alert('Error clearing metrics: ' + error.message);
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
    
    // Update grid limit when changed
    document.getElementById('grid-limit').addEventListener('change', updateGridVisualizations);

    // Initial status update
    updateStatus();
    updateMetrics();
    updateConsoleLogs();
    updateGridVisualizations();

    // Set up periodic updates
    updateInterval = setInterval(() => {
        updateStatus();
        updateConsoleLogs();  // Always update logs when training
        updateGridVisualizations();  // Always update grids
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

