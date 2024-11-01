

# Active RIS-Integrated TN-NTN Networks: DRL-Optimized Resource Allocation

This repository contains the code and resources for our work on **Deep Reinforcement Learning (DRL)-based optimization of resource allocation in Active Reconfigurable Intelligent Surface (A-RIS) integrated Terrestrial-Non-Terrestrial Networks (TN-NTN)**. By deploying a UAV-assisted active RIS along with a terrestrial RIS, our approach aims to improve sum-rate, energy efficiency, and user fairness in dynamic network environments. The optimization framework includes coordinated multipoint (CoMP) and non-orthogonal multiple access (NOMA) techniques, as well as a hybrid Proximal Policy Optimization (H-PPO) algorithm.

## Contents

- [System Model](#system-model)
- [Deep Reinforcement Learning Solution](#deep-reinforcement-learning-solution)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## System Model

Our system model comprises an **active RIS-assisted CoMP-NOMA downlink network** using single-input-single-output (SISO) transmission, with a UAV-based RIS alongside a fixed terrestrial RIS. The active RIS amplifies incoming signals, and phase shifts are optimized to achieve efficient communication.

Key components:
- **UAV-assisted and terrestrial RIS**: The active RIS at both fixed and mobile (UAV) locations are designed to amplify and redirect signals to users.
- **Dynamic UAV trajectory planning**: UAV positions are dynamically adjusted to maximize sum-rate while considering no-fly zones.
- **NOMA user pairing**: Users are grouped for diversity gains, enhancing the service to cell-edge users.
  
## Deep Reinforcement Learning Solution

The core of our solution uses **hybrid Proximal Policy Optimization (H-PPO)** to jointly optimize:
- **Power allocation and phase shifts** for active RIS.
- **UAV trajectory** for efficient coverage of mobile users.
- **NOMA-based user pairing** to improve edge-user experience and network performance.

The DRL model is formulated as an MDP (Markov Decision Process) where states include UAV position, power allocation, and RIS configurations. Actions involve updates to the RIS phase shifts, UAV movement, and BS power allocation, while the reward function incentivizes high sum rates and minimizes outage probabilities.

## Repository Structure

```
.
├── network/                     # Code for defining the neural networks (PPO, VAE, etc.)
├── params/                      # Parameter configurations for DRL and PPO training
├── LICENSE                      # License information
└── README.md                    # This readme file
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ahmd-mohsin/active-ris-tn-ntn-drl.git
cd active-ris-tn-ntn-drl
pip install -r requirements.txt
```

## Usage

To start training or run simulations, refer to the specific parameter files in the `params/` directory, which contain model configurations. Adjust the network structure as needed in the `network/` folder.

```bash
python train_ris_allocation.py --config params/your_config.yaml
```

## Results

Our experiments demonstrate that the **Active RIS with H-PPO** approach:
- Achieves up to 33% improvement in sum-rate over passive RIS setups.
- Reduces energy consumption while maintaining optimal signal-to-noise ratios.
- Increases user fairness and minimizes outage probability for edge users in dynamic TN-NTN environments.

## License

This project is licensed under the MIT License.

---
