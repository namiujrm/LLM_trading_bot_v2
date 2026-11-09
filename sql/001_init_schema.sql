-- ============================================================================
-- 001_init_schema.sql
-- Database Schema for LLM_trading_bot_v2
-- PostgreSQL 16 + TimescaleDB + pgvector
-- ============================================================================

-- Kích hoạt cả 2 extensions cốt lõi trên cùng 1 database
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- ============================================================================
-- BẢNG 1: TẦNG NỀN NẾN 1M ENRICHED (market_klines_1m_enriched)
-- Hypertable chunk 7 days, columnar compression on symbol, order by open_time DESC
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_klines_1m_enriched (
    open_time               TIMESTAMPTZ NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    open                    DOUBLE PRECISION NOT NULL,
    high                    DOUBLE PRECISION NOT NULL,
    low                     DOUBLE PRECISION NOT NULL,
    close                   DOUBLE PRECISION NOT NULL,
    volume                  DOUBLE PRECISION NOT NULL,
    quote_volume            DOUBLE PRECISION NOT NULL,
    taker_buy_vol           DOUBLE PRECISION NOT NULL,
    taker_buy_quote_vol     DOUBLE PRECISION NOT NULL,
    trade_count             INTEGER NOT NULL,
    -- Vi cấu trúc cộng dồn từ WebSocket aggTrade, depth20@100ms & forceOrder
    ofi_sum                 DOUBLE PRECISION NOT NULL, -- Order Flow Imbalance tích lũy trong 1m
    obi_mean                DOUBLE PRECISION NOT NULL, -- Trung bình Order Book Imbalance (-1..+1)
    obi_std                 DOUBLE PRECISION NOT NULL, -- Độ lệch chuẩn OBI trong 1m
    spread_bps_mean         DOUBLE PRECISION NOT NULL, -- Bid-Ask Spread trung bình (basis points)
    liq_long_usd            DOUBLE PRECISION NOT NULL, -- Thanh lý Long (USD)
    liq_short_usd           DOUBLE PRECISION NOT NULL, -- Thanh lý Short (USD)
    PRIMARY KEY (symbol, open_time)
);

-- Chuyển thành TimescaleDB Hypertable (chunk 7 ngày)
SELECT create_hypertable(
    'market_klines_1m_enriched',
    'open_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Thiết lập chính sách nén cột tự động
ALTER TABLE market_klines_1m_enriched SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'open_time DESC'
);

SELECT add_compression_policy(
    'market_klines_1m_enriched',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- CONTINUOUS AGGREGATES CHO market_klines_1m_enriched (15m, 30m, 1h, 4h)
-- ============================================================================

-- 1. Continuous Aggregate 15m
CREATE MATERIALIZED VIEW IF NOT EXISTS market_klines_15m_cagg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', open_time) AS bucket,
    symbol,
    first(open, open_time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, open_time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(taker_buy_vol) AS taker_buy_vol,
    sum(taker_buy_quote_vol) AS taker_buy_quote_vol,
    sum(trade_count) AS trade_count,
    sum(ofi_sum) AS ofi_sum,
    avg(obi_mean) AS obi_mean,
    avg(spread_bps_mean) AS spread_bps_mean,
    sum(liq_long_usd) AS liq_long_usd,
    sum(liq_short_usd) AS liq_short_usd
FROM market_klines_1m_enriched
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_klines_15m_cagg',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- 2. Continuous Aggregate 30m
CREATE MATERIALIZED VIEW IF NOT EXISTS market_klines_30m_cagg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 minutes', open_time) AS bucket,
    symbol,
    first(open, open_time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, open_time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(taker_buy_vol) AS taker_buy_vol,
    sum(taker_buy_quote_vol) AS taker_buy_quote_vol,
    sum(trade_count) AS trade_count,
    sum(ofi_sum) AS ofi_sum,
    avg(obi_mean) AS obi_mean,
    avg(spread_bps_mean) AS spread_bps_mean,
    sum(liq_long_usd) AS liq_long_usd,
    sum(liq_short_usd) AS liq_short_usd
FROM market_klines_1m_enriched
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_klines_30m_cagg',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- 3. Continuous Aggregate 1h
CREATE MATERIALIZED VIEW IF NOT EXISTS market_klines_1h_cagg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', open_time) AS bucket,
    symbol,
    first(open, open_time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, open_time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(taker_buy_vol) AS taker_buy_vol,
    sum(taker_buy_quote_vol) AS taker_buy_quote_vol,
    sum(trade_count) AS trade_count,
    sum(ofi_sum) AS ofi_sum,
    avg(obi_mean) AS obi_mean,
    avg(spread_bps_mean) AS spread_bps_mean,
    sum(liq_long_usd) AS liq_long_usd,
    sum(liq_short_usd) AS liq_short_usd
FROM market_klines_1m_enriched
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_klines_1h_cagg',
    start_offset => INTERVAL '14 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- 4. Continuous Aggregate 4h
CREATE MATERIALIZED VIEW IF NOT EXISTS market_klines_4h_cagg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('4 hours', open_time) AS bucket,
    symbol,
    first(open, open_time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, open_time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(taker_buy_vol) AS taker_buy_vol,
    sum(taker_buy_quote_vol) AS taker_buy_quote_vol,
    sum(trade_count) AS trade_count,
    sum(ofi_sum) AS ofi_sum,
    avg(obi_mean) AS obi_mean,
    avg(spread_bps_mean) AS spread_bps_mean,
    sum(liq_long_usd) AS liq_long_usd,
    sum(liq_short_usd) AS liq_short_usd
FROM market_klines_1m_enriched
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_klines_4h_cagg',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- ============================================================================
-- BẢNG 2: SNAPSHOT ĐẶC TRƯNG ĐỊNH LƯỢNG & ML 30M (quant_feature_vectors_30m)
-- Hypertable chunk 30 days
-- ============================================================================
CREATE TABLE IF NOT EXISTS quant_feature_vectors_30m (
    bar_time                TIMESTAMPTZ NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    regime_label            VARCHAR(32) NOT NULL,
    hurst_exp               DOUBLE PRECISION NOT NULL,
    yang_zhang_vol          DOUBLE PRECISION NOT NULL,
    vpin_value              DOUBLE PRECISION NOT NULL,
    kyle_lambda             DOUBLE PRECISION NOT NULL,
    oi_quadrant             VARCHAR(24) NOT NULL,
    ood_mahalanobis_dist    DOUBLE PRECISION NOT NULL,
    is_ood_warning          BOOLEAN NOT NULL,
    -- Dự báo Dual-Binary Causal GBDT
    p_long_win_cal          DOUBLE PRECISION NOT NULL,
    p_long_uncertainty      DOUBLE PRECISION NOT NULL,
    p_short_win_cal         DOUBLE PRECISION NOT NULL,
    p_short_uncertainty     DOUBLE PRECISION NOT NULL,
    model_version           VARCHAR(64) NOT NULL,
    features_json           JSONB NOT NULL,
    shap_explanation_json   JSONB NOT NULL,
    -- Stage 4 Causal ML Scalar Projections & Metrics
    gamma_ood               DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    ood_threshold           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    is_economic_consistent  BOOLEAN          NOT NULL DEFAULT TRUE,
    p_long_raw              DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    p_short_raw             DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    is_coherent             BOOLEAN          NOT NULL DEFAULT TRUE,
    coherence_penalty       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (symbol, bar_time)
);

SELECT create_hypertable(
    'quant_feature_vectors_30m',
    'bar_time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_quant_features_shap_gin
    ON quant_feature_vectors_30m USING gin (shap_explanation_json);

CREATE INDEX IF NOT EXISTS idx_quant_features_ood_drift
    ON quant_feature_vectors_30m (symbol, is_ood_warning, gamma_ood, bar_time DESC);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        ALTER TABLE quant_feature_vectors_30m SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'bar_time DESC'
        );

        PERFORM add_compression_policy(
            'quant_feature_vectors_30m',
            INTERVAL '30 days',
            if_not_exists => TRUE
        );
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Compression policy could not be attached: %', SQLERRM;
END $$;

-- ============================================================================
-- BẢNG 3: LỆNH THẬT, PAPER & VỊ THẾ ACTIVE (orders_and_positions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS orders_and_positions (
    order_id                UUID PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL,
    bar_time_30m            TIMESTAMPTZ NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    execution_mode          VARCHAR(16) NOT NULL, -- 'LIVE', 'TESTNET', 'PAPER'
    playbook_used           VARCHAR(40) NOT NULL,
    side                    VARCHAR(8)  NOT NULL, -- 'LONG', 'SHORT'
    entry_type              VARCHAR(24) NOT NULL, -- 'MARKET', 'LIMIT_PULLBACK_15M'
    entry_price             DOUBLE PRECISION NOT NULL,
    stop_loss_price         DOUBLE PRECISION NOT NULL,
    take_profit_price       DOUBLE PRECISION NOT NULL,
    vertical_barrier_time   TIMESTAMPTZ NOT NULL,
    notional_usd            DOUBLE PRECISION NOT NULL,
    leverage                INTEGER NOT NULL,
    isolated_margin_usd     DOUBLE PRECISION NOT NULL,
    is_sl_at_breakeven      BOOLEAN NOT NULL,
    effective_at_risk_usd   DOUBLE PRECISION NOT NULL,
    account_context_json    JSONB NOT NULL,
    status                  VARCHAR(24) NOT NULL, -- 'OPEN', 'FILLED', 'CLOSED_TP', 'CLOSED_SL', 'CLOSED_TIMEOUT'
    closed_at               TIMESTAMPTZ,
    exit_price              DOUBLE PRECISION,
    net_pnl_pct             DOUBLE PRECISION,
    net_pnl_usd             DOUBLE PRECISION,
    mae_pct                 DOUBLE PRECISION NOT NULL,
    mfe_pct                 DOUBLE PRECISION NOT NULL,
    agent_debate_json       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_pos_status_sym
    ON orders_and_positions (status, symbol, created_at DESC);

-- ============================================================================
-- BẢNG 4: VÒNG ĐỜI LỆNH ẢO SHADOW ORDERS & AUTOPSY (shadow_orders_lifecycle)
-- ============================================================================
CREATE TABLE IF NOT EXISTS shadow_orders_lifecycle (
    shadow_id               UUID PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL,
    bar_time_30m            TIMESTAMPTZ NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    playbook_used           VARCHAR(40) NOT NULL,
    veto_agent              VARCHAR(32) NOT NULL, -- 'Arbiter', 'RiskManager', 'OODDetector'
    veto_category           VARCHAR(32),          -- 'SETUP_DEFECT', 'PORTFOLIO_CAPACITY_VETO'
    veto_reason             TEXT NOT NULL,
    hypothetical_side       VARCHAR(8)  NOT NULL, -- 'LONG', 'SHORT'
    hypothetical_entry_price DOUBLE PRECISION NOT NULL,
    hypothetical_sl_price   DOUBLE PRECISION NOT NULL,
    hypothetical_tp_price   DOUBLE PRECISION NOT NULL,
    vertical_barrier_time   TIMESTAMPTZ NOT NULL,
    account_context_json    JSONB NOT NULL,
    status                  VARCHAR(24) NOT NULL, -- 'TRACKING', 'EXPIRED', 'HIT_TP_FIRST', 'HIT_SL_FIRST', 'STOP_HUNTED_THEN_HIT_TP'
    closed_at               TIMESTAMPTZ,
    hypothetical_net_pnl_pct DOUBLE PRECISION,
    mae_pct                 DOUBLE PRECISION NOT NULL,
    mfe_pct                 DOUBLE PRECISION NOT NULL,
    did_hit_tp_after_sl     BOOLEAN NOT NULL,
    post_sl_max_favorable   DOUBLE PRECISION,
    agent_debate_json       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shadow_status_sym
    ON shadow_orders_lifecycle (status, symbol, created_at DESC);

-- ============================================================================
-- BẢNG 5: BỘ NHỚ BÀI HỌC CA BỆNH 2 TẦNG TRÊN PGVECTOR (episodic_lessons_memory)
-- ============================================================================
CREATE TABLE IF NOT EXISTS episodic_lessons_memory (
    lesson_id               UUID PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL,
    updated_at              TIMESTAMPTZ NOT NULL,
    symbol                  VARCHAR(16) NOT NULL,
    playbook_used           VARCHAR(40) NOT NULL,
    side                    VARCHAR(8)  NOT NULL,
    outcome_category        VARCHAR(40) NOT NULL, -- 'VALIDATED_WIN', 'FALSE_POSITIVE_TRAP', 'STOP_HUNTED_THEN_HIT_TP', 'MISSED_OPPORTUNITY', 'VALIDATED_DEFENSE'
    confidence_score        DOUBLE PRECISION NOT NULL,
    is_archived             BOOLEAN NOT NULL,
    regime_trajectory_vec   vector(16) NOT NULL,
    realized_r_multiple     DOUBLE PRECISION NOT NULL,
    mae_atr_multiple        DOUBLE PRECISION NOT NULL,
    mfe_atr_multiple        DOUBLE PRECISION NOT NULL,
    did_hit_tp_after_sl     BOOLEAN NOT NULL,
    tier_1_index_json       JSONB NOT NULL,
    tier_2_dossier_json     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_lessons_vec_hnsw
    ON episodic_lessons_memory USING hnsw (regime_trajectory_vec vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_episodic_lessons_active
    ON episodic_lessons_memory (is_archived, symbol, outcome_category, confidence_score DESC);

-- ============================================================================
-- BẢNG 6: LỊCH SỬ TRẠNG THÁI TÀI KHOẢN, MARGIN & DRAWDOWN (account_equity_snapshots)
-- Phục vụ lắp ráp AccountPortfolioContext cho Agent 3 (Risk Arbiter) & Kiểm toán Drawdown
-- ============================================================================
CREATE TABLE IF NOT EXISTS account_equity_snapshots (
    snapshot_time           TIMESTAMPTZ NOT NULL,
    execution_mode          VARCHAR(16) NOT NULL, -- 'LIVE', 'TESTNET', 'PAPER'
    wallet_balance_usd      DOUBLE PRECISION NOT NULL,
    unrealized_pnl_usd      DOUBLE PRECISION NOT NULL,
    equity_nav_usd          DOUBLE PRECISION NOT NULL,
    used_margin_usd         DOUBLE PRECISION NOT NULL,
    available_margin_usd    DOUBLE PRECISION NOT NULL,
    margin_utilization_pct  DOUBLE PRECISION NOT NULL,
    open_positions_count    INTEGER NOT NULL,
    raw_open_heat_pct       DOUBLE PRECISION NOT NULL,
    effective_open_heat_pct DOUBLE PRECISION NOT NULL,
    net_btc_beta_usd        DOUBLE PRECISION NOT NULL,
    daily_dd_pct            DOUBLE PRECISION NOT NULL,
    max_dd_pct              DOUBLE PRECISION NOT NULL,
    consecutive_losses      INTEGER NOT NULL,
    positions_snapshot_json JSONB NOT NULL,
    PRIMARY KEY (execution_mode, snapshot_time)
);

SELECT create_hypertable(
    'account_equity_snapshots',
    'snapshot_time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- COMPATIBILITY ALIAS VIEWS (For System Design Doc 02 compatibility)
-- ============================================================================
CREATE OR REPLACE VIEW market_klines_1m_micro AS
    SELECT * FROM market_klines_1m_enriched;

CREATE OR REPLACE VIEW episodic_cases AS
    SELECT
        lesson_id AS case_id,
        created_at,
        updated_at,
        symbol,
        playbook_used,
        side,
        outcome_category,
        confidence_score,
        is_archived,
        regime_trajectory_vec,
        realized_r_multiple,
        mae_atr_multiple,
        mfe_atr_multiple,
        did_hit_tp_after_sl,
        tier_1_index_json,
        tier_2_dossier_json
    FROM episodic_lessons_memory;
