import yaml
import numpy as np

class CostEngine:
    def __init__(self, config_path='configs/cost_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.rto_logistics_cost = self.config['rto_logistics_cost']
        self.average_margin_pct = self.config['average_margin_pct']
        self.interventions = self.config['interventions']
        
    def evaluate_interventions(self, order_value, p_rto):
        margin = order_value * self.average_margin_pct
        
        results = {}
        for action, params in self.interventions.items():
            friction_cost = params['friction_cost']
            success_drop_pct = params['success_drop_pct']
            rto_reduction_pct = params['rto_reduction_pct']
            
            p_success = (1.0 - p_rto) * (1.0 - success_drop_pct)
            p_rto_after = p_rto * (1.0 - rto_reduction_pct)
            
            expected_loss = friction_cost + (p_rto_after * self.rto_logistics_cost) - (p_success * margin)
            results[action] = expected_loss
            
        return results
        
    def get_optimal_policy(self, df, proba_series):
        """
        Returns a tuple: (actions, expected_losses)
        Apply menu ONLY to payment_method == COD.
        Prepaid rows assigned 'PREPAID_PASSTHROUGH', 0 expected loss.
        """
        actions = []
        expected_losses = []
        
        for i, row in df.iterrows():
            payment_method = row['payment_method']
            if payment_method != "COD":
                actions.append("PREPAID_PASSTHROUGH")
                expected_losses.append(0.0)
                continue
                
            order_value = row['order_value']
            p_rto = proba_series.iloc[i]
            
            intervention_losses = self.evaluate_interventions(order_value, p_rto)
            best_action = min(intervention_losses, key=intervention_losses.get)
            best_loss = intervention_losses[best_action]
            
            actions.append(best_action)
            expected_losses.append(best_loss)
            
        return np.array(actions), np.array(expected_losses)
