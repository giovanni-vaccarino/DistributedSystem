import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flwr as fl
import argparse
from model.model import Model
from data.load_data import load_dataset
from shared.logger import setup_logger

logger = setup_logger(__name__)

class FlwClient(fl.client.NumPyClient):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        input_size = load_dataset(self.csv_path)[0].shape[1]
        self.model = Model(input_size)
        logger.info(f"Initialized client. Reading from csv: {csv_path}")

    def get_parameters(self, config):
        logger.info("Getting model parameters, to send it to the server")
        return self.model.get_weights()

    def fit(self, parameters, config):
        logger.info("Starting local training round")
        self.model.set_weights(parameters)
        # Reading always from a regularly updated csv
        X_train, _, y_train, _ = load_dataset(self.csv_path)

        # decide if it is better to call these hyperparam by server
        epochs = config.get("local_epochs", 8)
        batch_size = config.get("batch_size", 8)
        logger.info(f"Training with {len(X_train)} samples for {epochs} epochs (batch size {batch_size})")

        self.model.fit(X_train, y_train, epochs, batch_size)

        logger.info("Training complete. Sending updated weights to server")
        return self.model.get_weights(), len(X_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        _, X_test, _, y_test = load_dataset(self.csv_path)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        logger.info(f"Evaluation results: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(X_test), {"accuracy": accuracy}

def start_flower_client(server_address: str):
    csv_path = "../data/dataset_name.csv"
    client = FlwClient(csv_path)
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a Flower federated client.")
    parser.add_argument("--server_address", type=str, required=True)
    args = parser.parse_args()

    start_flower_client(server_address=args.server_address)
