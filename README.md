# PPO_improvement
CS294 final project

## Project Description
- Policy Gradients with Optimistic Value Functions  
- John Schulman  
- Policy gradient methods use value functions for variance reduction (e.g., see A3C or GAE). To obtain unbiased gradient estimates, the value function is chosen to approximate V^{\pi}, the value function of the current policy. There is reason to believe that we would obtain faster learning on many problems by instead using a value function that approximates V^*, the optimal value function. You can fit V^* by using Q-learning (to fit Q^*) or simply by fitting V to satisfy the inequality V(s) <= empirical return after state s rather than the equality V(s) = empirical return after state.
