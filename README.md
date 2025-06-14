# Reinforcement Learning — Four Rooms

## Overview

This project implements Q-learning agents in a 13×13 Four Rooms grid-world environment across three scenarios of increasing difficulty. The agent is trained to collect packages placed randomly on the grid. Each scenario represents a different learning challenge, including simple, multiple, and ordered package collection — with optional stochastic behavior.

## File Descriptions

### **1. `FourRooms.py`**
- The core environment file (provided).
- Models a 13x13 grid world divided into four rooms.
- Supports different package configurations and agent behaviors.
- Manages agent movement, environment state, terminal detection, and path visualization.

### **2. `Scenario1.py`**
- Implements Q-learning for **Scenario 1: Simple Package Collection**.
- Agent collects one package.
- Includes support for different ε-greedy strategies.
- Supports optional `--stochastic` flag for 20% random action transitions.
- Saves and displays the final path.

### **3. `Scenario2.py`**
- Implements Q-learning for **Scenario 2: Multiple Package Collection**.
- Agent must collect 3 packages in any order.
- Supports stochastic flag and saves final path.

### **4. `Scenario3.py`**
- Implements Q-learning for **Scenario 3: Ordered Package Collection**.
- Agent must collect packages in strict order: **Red → Green → Blue**.
- If a package is collected out of order, the episode ends immediately.

### **5. `requirements.txt`**
- Lists required libraries for running the environment and visualizations:
  - `numpy`
  - `matplotlib`
- Install dependencies using:
```bash
pip install -r requirements.txt
```

### **6. `README.md`**
- Provides an overview of the project.
- Describes each file’s purpose and how they integrate.
- Offers usage instructions for running each scenario.


## Usage Instructions

### **Running a Scenario**
Each scenario can be executed via command line. Use the `--stochastic` or `-s` flag to enable random action behavior.

**Scenario X**
```bash
python ScenarioX.py --stochastic
python ScenarioX.py -s
```
```bash
python ScenarioX.py
```

## Troubleshooting
- Make sure all files are in the same directory.
- Install required packages using `requirements.txt` before running.
- Use a terminal that supports matplotlib for proper visualization.

---

**Author:** Chiko Kasongo  
**Date:** 22/05/2025
