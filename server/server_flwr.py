import flwr as fl
from config import SERVER_ADDRESS, NUM_ROUNDS
from strategy import get_strategy

def main():
    strategy = get_strategy()

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
