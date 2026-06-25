import argparse

from edrft.wave import run_wave_experiment


def parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seeds(value):
    return tuple(int(item) for item in parse_csv(value))


def main():
    parser = argparse.ArgumentParser(description="Run RFT/edRFT wave forecasting experiments.")
    parser.add_argument("--data-dir", default="wave")
    parser.add_argument("--stations", default="46001h")
    parser.add_argument("--years", default="2017")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--look-back", type=int, default=48)
    parser.add_argument("--order", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=100)
    args = parser.parse_args()

    results = run_wave_experiment(
        data_dir=args.data_dir,
        stations=parse_csv(args.stations),
        years=parse_csv(args.years),
        seeds=parse_seeds(args.seeds),
        look_back=args.order if args.order is not None else args.look_back,
        horizon=args.horizon,
        n_layers=args.layers,
        max_evals=args.max_evals,
    )
    for result in results:
        print(
            f"{result.year} {result.station} seed={result.seed} {result.model:5s} "
            f"RMSE={result.rmse:.6f} MAPE={result.mape:.6f} MASE={result.mase:.6f} "
            f"tune={result.tuning_seconds:.3f}s train={result.training_seconds:.3f}s "
            f"test={result.testing_seconds:.3f}s"
        )
        print(f"best_params={result.best_params}")


if __name__ == "__main__":
    main()
