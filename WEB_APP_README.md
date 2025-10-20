# ARC Training Web Interface

A web-based dashboard for managing and monitoring DreamerV3 ARC training sessions.

## Quick Start

1. **Install Flask** (if not already installed):
   ```bash
   pip install flask
   ```

2. **Start the Web App**:
   ```bash
   cd dreamerv3
   python app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

## Features

### Training Control
- **Start Training**: Select a configuration and optionally add custom arguments
- **Stop Training**: Gracefully stop the current training session
- **Clear Data**: Reset all accumulated metrics

### Available Configurations
- `arc` - Default ARC V2 Training
- `arc_v2_train` - ARC V2 Training Set
- `arc_v2_eval` - ARC V2 Evaluation Set
- `arc_v1_train` - ARC V1 Training Set
- `arc_v1_eval` - ARC V1 Evaluation Set

### Real-Time Monitoring

The dashboard automatically updates every 2 seconds with:

1. **Score (Accuracy) Chart**
   - Raw score values per episode
   - Moving average for trend analysis
   - Represents how accurately the agent solves puzzles (0.0 - 1.0)

2. **Episode Length Chart**
   - Number of steps per episode
   - Moving average for trend analysis

3. **FPS Chart**
   - Policy FPS (frames per second)
   - Moving average for performance monitoring

### Metrics Display

The interface shows both **raw values** and **moving averages** for all metrics:
- Current step number
- Latest score and moving average
- Episode length and moving average
- FPS and moving average

### Moving Average Window

You can adjust the moving average window size (default: 10) to control smoothing:
- Smaller values (e.g., 5): More responsive to recent changes
- Larger values (e.g., 50): Smoother trends, less noise

## How It Works

### Backend (`app.py`)
- Flask web server running on port 5000
- Spawns training subprocess when started
- Monitors `metrics.jsonl` file in the logdir
- Calculates moving averages in real-time
- Provides REST API endpoints for status and metrics

### Frontend
- Modern, responsive UI built with HTML/CSS/JavaScript
- Chart.js for interactive visualizations
- Auto-refreshing every 2 seconds
- Smooth animations and transitions

## API Endpoints

If you want to integrate with the backend programmatically:

- `GET /api/status` - Get training status
- `GET /api/metrics?window=10` - Get metrics with specified moving average window
- `POST /api/start` - Start training (JSON body: `{config, custom_args}`)
- `POST /api/stop` - Stop training
- `POST /api/clear` - Clear metrics data

## Training Logs

Training logs are saved to `~/logdir/<timestamp>/`:
- `metrics.jsonl` - All training metrics
- `scores.jsonl` - Episode scores
- `ckpt/` - Model checkpoints
- `config.yaml` - Training configuration

## Tips

1. **For Quick Tests**: Use custom args like `--run.steps 1e5` to limit training steps
2. **Memory Management**: Clear data between runs if needed
3. **Multiple Sessions**: The app tracks the most recent training session
4. **Patience**: Training may take a few seconds to start and begin logging metrics

## Troubleshooting

**No metrics showing up?**
- Wait 30-60 seconds after starting training
- Check that the logdir path is displayed in the UI
- Ensure the ARC data is correctly placed in `arc-data/` directory

**Training won't start?**
- Make sure no other training process is running
- Check that all dependencies are installed
- Verify the `dreamerv3/main.py` file exists

**Web app won't start?**
- Ensure Flask is installed: `pip install flask`
- Check that port 5000 is not already in use
- Try running with: `python -u app.py` for unbuffered output

## Architecture

```
dreamerv3/
├── app.py                    # Flask backend
├── templates/
│   └── index.html           # Main web interface
└── static/
    ├── style.css            # Styling
    └── app.js               # Frontend JavaScript
```

The app is completely self-contained in the `dreamerv3` folder and requires no additional configuration.

## Credits

Built for DreamerV3 ARC training monitoring.
Uses Chart.js for visualizations and Flask for the backend.

