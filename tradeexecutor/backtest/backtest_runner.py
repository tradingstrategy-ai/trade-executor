from pathlib import Path


def run_backtest(
        strategy_path: Path,
        start_at: str,
        end_at: str,
    ):
    """High-level entry point for running a single backtest."""

    execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        execution_type,
        json_rpc,
        private_key,
        gas_price_method,
        confirmation_timeout,
        confirmation_block_count,
        max_slippage,
    )

