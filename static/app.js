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
    // Score Chart
    const scoreCtx = document.getElementById('score-chart').getContext('2d');
    charts.score = new Chart(scoreCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Score',
                    data: [],
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.1,
                    borderWidth: 1,
                    pointRadius: 2,
                },
                {
                    label: 'Score (Moving Avg)',
                    data: [],
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0,
                }
            ]
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

    // FPS Chart
    const fpsCtx = document.getElementById('fps-chart').getContext('2d');
    charts.fps = new Chart(fpsCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'FPS',
                    data: [],
                    borderColor: 'rgba(251, 146, 60, 0.5)',
                    backgroundColor: 'rgba(251, 146, 60, 0.1)',
                    tension: 0.1,
                    borderWidth: 1,
                    pointRadius: 2,
                },
                {
                    label: 'FPS (Moving Avg)',
                    data: [],
                    borderColor: 'rgba(251, 146, 60, 1)',
                    backgroundColor: 'rgba(251, 146, 60, 0.2)',
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

        // Score chart
        charts.score.data.labels = data.steps.slice(startIdx);
        charts.score.data.datasets[0].data = data.scores.slice(startIdx);
        charts.score.data.datasets[1].data = data.scores_ma.slice(startIdx);
        charts.score.update('none');

        // Length chart
        charts.length.data.labels = data.steps.slice(startIdx);
        charts.length.data.datasets[0].data = data.lengths.slice(startIdx);
        charts.length.data.datasets[1].data = data.lengths_ma.slice(startIdx);
        charts.length.update('none');

        // FPS chart
        charts.fps.data.labels = data.steps.slice(startIdx);
        charts.fps.data.datasets[0].data = data.fps.slice(startIdx);
        charts.fps.data.datasets[1].data = data.fps_ma.slice(startIdx);
        charts.fps.update('none');

        // Update latest values
        if (data.latest) {
            document.getElementById('latest-step').textContent = data.latest.step || 0;
            document.getElementById('latest-score').textContent = (data.latest.score || 0).toFixed(3);
            document.getElementById('latest-score-ma').textContent = (data.latest.score_ma || 0).toFixed(3);
            document.getElementById('latest-length').textContent = Math.round(data.latest.length || 0);
            document.getElementById('latest-length-ma').textContent = (data.latest.length_ma || 0).toFixed(1);
            document.getElementById('latest-fps').textContent = (data.latest.fps || 0).toFixed(1);
            document.getElementById('latest-fps-ma').textContent = (data.latest.fps_ma || 0).toFixed(1);
        }

    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
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

    // Initial status update
    updateStatus();
    updateMetrics();

    // Set up periodic updates
    updateInterval = setInterval(() => {
        updateStatus();
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

