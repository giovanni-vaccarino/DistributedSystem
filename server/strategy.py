from flwr.server.strategy import FedAvg
from config import MIN_CLIENTS, MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, LOCAL_EPOCHS, MODEL_DIR
import os
import numpy as np
import flwr as fl

os.makedirs(MODEL_DIR, exist_ok=True)

class SaveModelStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # Saving at each round the parameters (overwriting to don't use too memory)
        if aggregated_weights is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            file_path = os.path.join(MODEL_DIR, f"global_model.npy")
            np.savez(file_path, *weights)
            print(f"[Server] Saved global model at round {server_round} to {file_path}")

        return aggregated_weights, metrics_aggregated
    def aggregate_evaluate(self, server_round, results, failures):
   	 aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

   	 # Custom aggregation of accuracy
   	 accuracies = []
   	 num_examples = []

   	 for _, eval_res in results:
       		 if "accuracy" in eval_res.metrics:
           		 accuracies.append(eval_res.metrics["accuracy"])
           		 num_examples.append(eval_res.num_examples)

   	 if accuracies:
       	 # Weighted average of accuracy
       		 total_examples = sum(num_examples)
       		 weighted_accuracy = sum(acc * n for acc, n in zip(accuracies, num_examples)) / total_examples
        	 print(f"[Server] Round {server_round} - Aggregated accuracy: {weighted_accuracy:.4f}")
   	 else:
       		 weighted_accuracy = None

   	 return aggregated_loss, {"accuracy": weighted_accuracy}


def get_strategy():
    return SaveModelStrategy(
        fraction_fit=1.0, # When trianing sample all the available clients
        fraction_evaluate=1.0, # When evaluating sample all the available clients
        min_available_clients=MIN_CLIENTS,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        on_fit_config_fn=lambda rnd: {"epochs": LOCAL_EPOCHS},
    )

