# CARLA Training Data Generator

## Overview
This repository contains scripts for generating training data from the CARLA autonomous driving simulator. The data is used to train neural networks for autonomous vehicle control, including steering, throttle, and brake decisions based on camera inputs and vehicle state information.

## Project Structure
```
.
├── scripts/           # Data collection and processing scripts
├── models/           # Trained model weights and architectures
├── data/             # Generated training data
│   ├── raw/         # Raw simulator outputs
│   └── processed/   # Processed and formatted training data
└── configs/         # Configuration files for data collection
```

## Requirements
- CARLA Simulator (0.9.13)
- carla
- Python 3.6.9
- Required Python packages:
  - carla 0.9.13
  - numpy
  - opencv-python
  - pandas
  - tqdm

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/carla-driver-data.git
cd carla-driver-data
```

2. Run the setup script to create the required directory structure:
```bash
./setup.sh
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the CARLA simulator:
```bash
./CarlaUE4.sh
```

2. Run the data collection script:
```bash
python scripts/collect_data.py --episodes 100 --weather random
```

3. Process the collected data:
```bash
python scripts/process_data.py --input data/raw --output data/processed
```

## Data Collection
The data collection process captures:
- Front-facing camera images (RGB)
- Vehicle telemetry (speed, acceleration, location)
- Control inputs (steering, throttle, brake)
- Environmental conditions (weather, time of day)
- Traffic information (surrounding vehicles, pedestrians)

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
See the LICENSE file for details.
