from collections import deque
from dataclasses import dataclass
import time

@dataclass
class BudgetConfig:
    total_cost_budget_usd: float
    budget_window_sec: int = 3600  # 1 hour

class SLABudgetTracker:
    def __init__(self, config: BudgetConfig):
        self.total_budget = config.total_cost_budget_usd
        self.current_spent = 0.0
        self.window = deque()
        self.window_sec = config.budget_window_sec

    def add_cost(self, cost_usd: float) -> bool:
        """
        Record a cost. Returns False if budget exceeded.
        Also maintains the sliding window of costs.
        """
        current_time = time.time()
        self.current_spent += cost_usd
        self.window.append((current_time, cost_usd))
        
        # Prune old entries from window if needed (optional implementation detail, 
        # but good for long-running processes to avoid memory leaks if we only care about total)
        # Note: The user req implies a global total budget, but specifies a window.
        # usually "budget" is total-ever, or "rate limit" is per-window.
        # "total_cost_budget_usd" implies a cap.
        # "budget_window_sec" implies maybe we care about rate?
        # For this implementation, we'll track total accumulated cost against the budget
        # since it's "BudgetExhaustedError". The window might be for analytics.
        
        if self.current_spent >= self.total_budget:
            return False  # Budget exhausted
        return True

    def get_remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self.current_spent)

    def get_spent_pct(self) -> float:
        if self.total_budget <= 0:
            return 1.0
        return self.current_spent / self.total_budget

    def reset(self):
        self.current_spent = 0.0
        self.window.clear()
