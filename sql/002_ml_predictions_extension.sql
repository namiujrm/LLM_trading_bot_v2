-- ============================================================================
-- 002_ml_predictions_extension.sql
-- Stage 4 Causal ML Predictions & SHAP Extension for quant_feature_vectors_30m
-- ============================================================================

-- 1. Thêm các cột vô hướng của Stage 4 ML phục vụ truy vấn và lọc nhanh
ALTER TABLE quant_feature_vectors_30m
    ADD COLUMN IF NOT EXISTS gamma_ood              DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS ood_threshold          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS is_economic_consistent  BOOLEAN          NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS p_long_raw             DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS p_short_raw            DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS is_coherent            BOOLEAN          NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS coherence_penalty      DOUBLE PRECISION NOT NULL DEFAULT 0.0;

-- 2. GIN Index trên shap_explanation_json để tối ưu hóa truy vấn JSONB sâu
CREATE INDEX IF NOT EXISTS idx_quant_features_shap_gin
    ON quant_feature_vectors_30m USING gin (shap_explanation_json);

-- 3. Chỉ mục B-Tree phục vụ bộ lọc OOD & Calibration Drift
CREATE INDEX IF NOT EXISTS idx_quant_features_ood_drift
    ON quant_feature_vectors_30m (symbol, is_ood_warning, gamma_ood, bar_time DESC);

-- 4. Kích hoạt TimescaleDB columnar compression cho quant_feature_vectors_30m (nếu có TimescaleDB)
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
