# Adapters Layer Rules (`src/trading_bot/adapters/`)

1. **Sub-Adapter Isolation (Contract 4):**
   - Sub-adapters (e.g., `persistence`, `binance`, `telegram`) must remain completely isolated from one another.
   - Calling or importing between sub-adapters is strictly prohibited (`[tool.importlinter.contracts.adapters_isolation]`).

2. **Port Implementation Contract:**
   - Every adapter must implement the exact Protocol / ABC defined in `src/trading_bot/ports/`.
   - Never expose vendor-specific exceptions or models directly across layer boundaries; translate them to typed Domain exceptions/entities.

3. **Explicit Configuration Injection:**
   - Every adapter must receive its connection settings, credentials, and parameters via explicit Config objects (e.g., `DatabaseConfig`, `RedisConfig`, `BinanceConfig`).
   - Default fallback arguments and `.get(key, default)` calls are strictly prohibited.
