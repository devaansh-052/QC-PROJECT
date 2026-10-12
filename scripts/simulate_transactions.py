import argparse
from quantum_fraud.inference.simulation import simulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send synthetic demonstration traffic; no real money")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(simulate(args.url, args.count, args.seed))
