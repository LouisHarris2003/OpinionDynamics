# OpinionDynamics

**Opinion Dynamics Simulation with Deffuant Model**

This Python script simulates opinion dynamics using the Deffuant model, allowing for both networked and non-networked scenarios. Below is a brief overview along with instructions on how to use the script.

**Usage:**

1. **Non-Networked Simulation:**
   - Run the script without any arguments to perform a non-networked simulation.
   - Use the `--defuant` flag to enable the non-networked simulation.
   - Optional arguments:
     - `--beta`: Set the weighting factor for updating opinions (default: 0.2).
     - `--threshold`: Set the threshold value for initiating interaction (default: 0.2).

2. **Networked Simulation:**
   - Use the `--defuant` flag along with `--use_network` to enable the networked simulation.
   - Provide the size of the network using the `--use_network` flag.
   - Optional arguments:
     - `--beta`: Set the weighting factor for updating opinions (default: 0.2).
     - `--threshold`: Set the threshold value for initiating interaction (default: 0.2).

3. **Testing:**
   - Use the `--test_defuant` flag to execute unit tests for the `defuant` function.
   - This ensures correct functionality of the simulation.

**Example Commands:**

1. Run non-networked simulation:
   ```
   python script.py --defuant
   ```

2. Run networked simulation with a small-world network of size 20:
   ```
   python script.py --defuant --use_network 20
   ```

3. Run unit tests:
   ```
   python script.py --test_defuant
   ```

**Dependencies:**
- NumPy
- Matplotlib
- unittest (for testing)

**Note:**
- The script includes visualization of opinion dynamics using histograms and plots.
- Adjust parameters such as `threshold`, `beta`, and `num_timesteps` to explore different scenarios.
- For networked simulations, the Watts-Strogatz small-world graph is utilized.
- Refer to the function docstrings for detailed information on each parameter and functionality.
