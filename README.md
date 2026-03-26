# 🎰 Multi-Armed Bandit: Explore-Then-Exploit (A/B Testing)

This project is a Streamlit web application that simulates the **Explore-then-Exploit** (A/B Testing) strategy for a classic Reinforcement Learning problem: the Multi-Armed Bandit.

## 📝 About the Simulation
In this scenario, we have a total budget (e.g., $10,000) to pull the levers of three different slot machines (Bandits A, B, and C), each with an unknown "true" win rate. 

The strategy is divided into two phases:
1. **Exploration Phase (A/B Test)**: We allocate a specific budget (e.g., $2,000) equally between Bandit A and Bandit B to figure out which one has a better empirical win rate.
2. **Exploitation Phase**: We take the winner of the A/B test and allocate the entire remaining budget (e.g., $8,000) solely to that bandit to maximize our return.

## 📊 Dashboard & Visualizations
This app provides a comprehensive dashboard with three main visualizations:
1. **Cumulative Average Return vs. Dollars Spent**: A line chart showing the overall return trajectory of the A/B testing strategy over time, compared to the Optimal Return.
2. **True vs. Estimated Bandit Means**: A clear bar chart comparing the mathematically defined true means side-by-side with the empirical means discovered during the exploration phase.
3. **Independent Convergence Simulation**: A detailed rolling-average chart tracking Bandits A, B, and C pulled independently, demonstrating how empirical averages progressively converge toward their true structural probabilities as the sample size increases.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jjys/MAB.git
   cd MAB
   ```

2. **Install the required dependencies:**
   Make sure you have Python 3.8+ installed. Then simply run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your default web browser at `http://localhost:8501`.

## ☁️ How to Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
2. Click **New app** > **Use existing repo**.
3. Select your repository (`jjys/MAB`), set the branch to `main`, and the Main file path to `app.py`.
4. Click **Deploy!** 

*(Note: We deliberately avoided strict version pinning in `requirements.txt` to ensure Streamlit's environment can fetch the latest compatible dependencies without compilation errors.)*

## 🛠 Built With
* [Streamlit](https://streamlit.io/) - Web framework for data apps
* [NumPy](https://numpy.org/) - Numerical computing for the Bernoulli bandit simulation
* [Matplotlib](https://matplotlib.org/) - Data visualization & plotting
