import pickle

class MarkovDecisionProcess:
    def __init__(self, states, actions, transitions, rewards, gamma = 0.9):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.V = {}
        self.Q = {}   
        for s in states:
            self.V[s] = 0.0
            
        self.policy = {}
        for state in states:
            if actions:
                self.policy[state] = actions[0]
            else:
                self.policy[state] = None

    def _compute_q_value(self, s, a):
        val = self.rewards.get(s, {}).get(a, 0)
        for prob, next_s in self.transitions[s][a]:
            val += self.gamma * prob * self.V.get(next_s, 0.0)
        return val

    def compute_q_table(self):
        self.Q = {}
        for s in self.states:
            if s not in self.transitions:
                continue
            self.Q[s] = {}
            for a in self.transitions[s]:
                self.Q[s][a] = self._compute_q_value(s, a)
        return self.Q

    def _extract_policy(self):
        self.compute_q_table()
        for s, q_s in self.Q.items():
            if not q_s:
                continue
            best_action = max(q_s, key=q_s.get)
            self.policy[s] = best_action

    def value_iteration(self, tolerance=1e-6, max_iter=10000):
        for i in range(max_iter):
            delta = 0
            new_V = self.V.copy()
            for s in self.states:
                if s not in self.transitions:
                    continue

                action_values = []
                for a in self.transitions[s]:
                    val = self.rewards.get(s, {}).get(a, 0)
                    for prob, next_s in self.transitions[s][a]:
                        val += self.gamma * prob * self.V[next_s]
                    action_values.append(val)
                
                if action_values:
                    best_val = max(action_values)
                    delta = max(delta, abs(new_V[s] - best_val))
                    new_V[s] = best_val
                 
            self.V = new_V
            if i % 10 == 0:  
                print(f"  VI iter {i}: delta = {delta:.6f}")
            if delta < tolerance:
                print(f"Value iteration converged in {i+1} iterations")
                break
        else:
            print(f"Reached max iterations ({max_iter}) without converging")

        self._extract_policy()
        return self.V
    
    def policy_iteration(self, tolerance=1e-6, max_iter=100, eval_max_iter=1000):
        for pi_iter in range(max_iter):
            # policy evaluation 
            for eval_iter in range(eval_max_iter):
                delta = 0
                new_V = self.V.copy()
                for s in self.states:
                    if s not in self.transitions:
                        continue
                    a = self.policy[s]
                    if a is None or a not in self.transitions[s]:
                        continue
                    val = self.rewards.get(s, {}).get(a, 0)
                    for prob, next_s in self.transitions[s][a]:
                        val += self.gamma * prob * self.V[next_s]
                    delta = max(delta, abs(new_V[s] - val))
                    new_V[s] = val
                self.V = new_V
                if delta < tolerance:
                    break

            self.compute_q_table()
            policy_stable = True
            for s in self.states:
                if s not in self.Q or not self.Q[s]:
                    continue
                old_action = self.policy[s]
                best_action = max(self.Q[s], key=self.Q[s].get)
                self.policy[s] = best_action
                if old_action != best_action:
                    policy_stable = False
            
            if policy_stable:
                print(f"Policy iteration converged in {pi_iter + 1} iterations")
                return self.V
                
        print(f"Policy iteration reached max iterations ({max_iter})")        
        return self.V

    def save_best(self, path="best_policy.pkl"):
        payload = {
            "policy": dict(self.policy),
            "V": dict(self.V),
            "Q": {s: dict(q_a) for s, q_a in self.Q.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved policy ({len(self.policy)} states) to {path}")

    def load_best(self, path="best_policy.pkl"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.policy = payload["policy"]
        self.V = payload["V"]
        self.Q = payload.get("Q", {})
        print(f"Loaded policy ({len(self.policy)} states) from {path}")