# Electricity Spot Price Prediction and Solar Panel Control

This project aims to predict electricity spot prices using an LSTM neural network and automatically switch off solar panels when prolonged negative prices are detected. The model predicts prices based on historical data, and if significant downward trends are detected (e.g., sustained negative prices), it sends signals to switch off solar panels to avoid negative revenue impact.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Modbus Communication](#modbus-communication)
- [Parameters](#parameters)
- [Results](#results)
- [Notes](#notes)

## Project Structure
- **`Forecasting.py`**: Main script for data processing, training, prediction, trend detection, and Modbus signaling.
- **`/unused/082022.csv`**, **`/unused/082023.csv`**, etc.: CSV files with historical electricity data (`SETTLEMENTDATE` and `RRP` columns) for model training.
- **`modbus_server`**: A simulated modbus_server using pymodbus
- **README.md**: Documentation for the project.

## Requirements
- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`
  - `keras`
  - `tensorflow`
  - `pymodbus`

## Installation
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare Data**: Place monthly electricity spot price data CSV files in the `/unused` directory, each containing at least `SETTLEMENTDATE` and `RRP` columns.

2. **Run the Script**:
   Start the server:
   ```bash
   python modbus_server.py
   ```

   
   Execute the main script:
   ```bash
   python Forecasting.py
   ```
   The script:
   - Loads and preprocesses data.
   - Trains the LSTM model and makes predictions for a specified 24-hour period.
   - Detects prolonged negative price trends.
   - Sends a Modbus signal to switch off solar panels during detected periods.

## Modbus Communication
The script uses the Modbus protocol to control solar panels via a Modbus TCP server. When a 2-hour negative price trend is detected, it sends a switch-off signal to the Modbus server, then re-enables solar panels after the specified period.

**Connection Setup**:
- Host: `localhost`
- Port: `5020`
- Coil Address: `0x01`

If `ModbusTcpClient` cannot connect, it prints an error and exits.

## Parameters
- `time_steps`: Time steps for LSTM sequence (288 for 24 hours).
- `min_duration`: Minimum duration for detecting prolonged downward trends (default: 2 hours, or 24 points).
- `drop_threshold`: RRP threshold for identifying significant downward trends.
- `bid_price_threshold`: Bidding price threshold used in plotting and control signals.

## Results
- **RMSE Calculation**: Computes RMSE between predicted and actual prices for evaluation.
- **Plotting**: Plots predicted vs. actual prices and indicates bidding prices. Plots use 5-minute intervals for a 24-hour period.

## Notes
- The current setup uses simulated data (`/unused/082022.csv`, `/unused/082023.csv`, etc.) for model training. Adjust paths and formats as needed for different datasets.
- Modbus operations are simplified here; actual deployments may need event-based scheduling for turning panels back on after each detected period.
