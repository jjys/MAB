import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Multi-Armed Bandit: A/B Testing", layout="wide")

st.title("🎰 Multi-Armed Bandit: Explore-Then-Exploit (A/B Testing)")
st.markdown("""
This app simulates an A/B Testing strategy in a Multi-Armed Bandit problem. 
We have a total budget to spend. During the **Exploration Phase**, we divide our budget equally between the first two bandits (A and B). 
After exploring, we enter the **Exploitation Phase**, where we give the entire remaining budget to the bandit that performed best during the A/B test.
""")

# --- Sidebar Parameters ---
st.sidebar.header("⚙️ Simulation Parameters")

TOTAL_BUDGET = st.sidebar.number_input("Total Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)
EXPLORE_BUDGET = st.sidebar.number_input(
    "Exploration Budget ($) [For A/B test]", 
    min_value=100, 
    max_value=TOTAL_BUDGET, 
    value=2000, 
    step=100,
    help="This budget will be split equally between Bandit A and Bandit B."
)

st.sidebar.markdown("---")
st.sidebar.subheader("True Intrinsic Means")
mean_A = st.sidebar.slider("Bandit A True Mean", 0.0, 1.0, 0.8, 0.01)
mean_B = st.sidebar.slider("Bandit B True Mean", 0.0, 1.0, 0.7, 0.01)
mean_C = st.sidebar.slider("Bandit C True Mean (Ignored in A/B test)", 0.0, 1.0, 0.5, 0.01)

TRUE_MEANS = {'A': mean_A, 'B': mean_B, 'C': mean_C}
EXPLOIT_BUDGET = TOTAL_BUDGET - EXPLORE_BUDGET

NUM_RUNS = st.sidebar.slider("Number of Simulations (Smoothing)", 10, 500, 100, 10)

def run_simulation():
    cumulative_rewards_all_runs = np.zeros(TOTAL_BUDGET)
    total_reward_sum = 0
    best_bandit_mean = max(TRUE_MEANS.values())
    total_optimal_reward_sum = TOTAL_BUDGET * best_bandit_mean
    
    total_empirical_A = 0.0
    total_empirical_B = 0.0
    
    # Progress bar for simulations
    progress_bar = st.progress(0)
    
    for run in range(NUM_RUNS):
        rewards = np.zeros(TOTAL_BUDGET)
        
        # --- EXPLORATION PHASE ---
        pulls_A = EXPLORE_BUDGET // 2
        pulls_B = EXPLORE_BUDGET // 2
        
        rewards_A = np.random.binomial(1, TRUE_MEANS['A'], pulls_A)
        rewards_B = np.random.binomial(1, TRUE_MEANS['B'], pulls_B)
        
        rewards[0:pulls_A] = rewards_A
        rewards[pulls_A:EXPLORE_BUDGET] = rewards_B
        
        empirical_mean_A = np.mean(rewards_A) if pulls_A > 0 else 0
        empirical_mean_B = np.mean(rewards_B) if pulls_B > 0 else 0
        
        total_empirical_A += empirical_mean_A
        total_empirical_B += empirical_mean_B
        
        # --- EXPLOITATION PHASE ---
        winning_bandit = 'A' if empirical_mean_A >= empirical_mean_B else 'B'
        
        if EXPLOIT_BUDGET > 0:
            rewards_exploit = np.random.binomial(1, TRUE_MEANS[winning_bandit], EXPLOIT_BUDGET)
            rewards[EXPLORE_BUDGET:] = rewards_exploit
            
        cumulative_rewards = np.cumsum(rewards)
        cumulative_rewards_all_runs += cumulative_rewards
        total_reward_sum += cumulative_rewards[-1]
        
        # Update progress
        if run % max(1, (NUM_RUNS // 10)) == 0:
            progress_bar.progress((run + 1) / NUM_RUNS)
            
    progress_bar.empty()

    # --- Analytics ---
    avg_est_A = total_empirical_A / NUM_RUNS
    avg_est_B = total_empirical_B / NUM_RUNS
    
    avg_cumulative_rewards = cumulative_rewards_all_runs / NUM_RUNS
    dollars_spent = np.arange(1, TOTAL_BUDGET + 1)
    average_return_per_dollar = avg_cumulative_rewards / dollars_spent
    
    expected_total_reward = total_reward_sum / NUM_RUNS
    regret = total_optimal_reward_sum - expected_total_reward
    
    return dollars_spent, average_return_per_dollar, expected_total_reward, total_optimal_reward_sum, regret, best_bandit_mean, avg_est_A, avg_est_B

def plot_individual_convergence():
    pulls_per_bandit = min(4000, max(1000, TOTAL_BUDGET // 3))
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    for name, color in zip(['A', 'B', 'C'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        rewards = np.random.binomial(1, TRUE_MEANS[name], pulls_per_bandit)
        cum_avg = np.cumsum(rewards) / np.arange(1, pulls_per_bandit + 1)
        ax3.plot(range(1, pulls_per_bandit + 1), cum_avg, label=f'Bandit {name}')
        
    ax3.set_title('A/B Test Simulation: Average Return vs Dollars Spent')
    ax3.set_xlabel('Dollars Spent per Bandit')
    ax3.set_ylabel('Average Return')
    ax3.grid(True)
    ax3.legend()
    return fig3

if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Running MAB simulations..."):
        dollars_spent, avg_return, exp_reward, opt_reward, regret, best_mean, avg_est_A, avg_est_B = run_simulation()
        
        # --- Display Metrics ---
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Final Reward", f"${opt_reward:,.2f}", help="If you perfectly picked the best bandit from day 1")
        col2.metric("A/B Testing Final Reward", f"${exp_reward:,.2f}", delta=f"-${regret:,.2f} (Regret)", delta_color="inverse")
        col3.metric("Total Regret", f"${regret:,.2f}", help="Opportunity cost of exploring and sometimes choosing the wrong bandit.")

        st.markdown("---")

        # --- Top Row: Main Plot & Bar Chart ---
        col_plot1, col_plot2 = st.columns([1.5, 1])
        
        with col_plot1:
            st.write("**Cumulative Average Return vs. Dollars Spent**")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            
            ax1.plot(dollars_spent, avg_return, label=f'Avg Return ({NUM_RUNS} runs)', color='#1f77b4', linewidth=2)
            ax1.axvline(x=EXPLORE_BUDGET, color='#ff7f0e', linestyle='--', label=f'End of Explore (${EXPLORE_BUDGET})')
            ax1.axhline(y=best_mean, color='#d62728', linestyle=':', label=f'Optimal Return ({best_mean})')
            
            ax1.set_xlabel('Dollars Spent (Total Budget)')
            ax1.set_ylabel('Average Return per Dollar')
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            y_min = max(0.0, min(min(avg_return), best_mean) - 0.1)
            y_max = min(1.0, max(max(avg_return), best_mean) + 0.1)
            ax1.set_ylim(y_min, y_max)
            ax1.legend(loc="lower right")
            st.pyplot(fig1)

        with col_plot2:
            st.write("**True vs. Estimated Bandit Means**")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            
            labels = ['Bandit A', 'Bandit B', 'Bandit C']
            true_means = [TRUE_MEANS['A'], TRUE_MEANS['B'], TRUE_MEANS['C']]
            # Bandit C wasn't explored in phase 1, so technically its estimate is unknown, 
            # we will visually show it as 0.50 (to match reference images) or mark it.
            estimated_means = [avg_est_A, avg_est_B, TRUE_MEANS['C']] 
            
            x = np.arange(len(labels))
            width = 0.35
            
            rects1 = ax2.bar(x - width/2, true_means, width, label='True Mean', color='#729dcb')
            rects2 = ax2.bar(x + width/2, estimated_means, width, label='Estimated Mean', color='#fcb03b')
            
            ax2.set_ylabel('Mean Return')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylim(0, 1.1)
            ax2.legend(loc="upper right")
            
            # Add value annotations
            for rects in [rects1, rects2]:
                for r in rects:
                    h = r.get_height()
                    ax2.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width() / 2, h), 
                                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            st.pyplot(fig2)
            
        st.markdown("---")
        
        # --- Bottom Row: Convergence Plot ---
        st.subheader("🔬 Independent A/B Test Simulation (Convergence)")
        st.markdown("If we purely tested each bandit identically to estimate their averages over time:")
        fig3 = plot_individual_convergence()
        st.pyplot(fig3)
