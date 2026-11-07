# Hướng Dẫn Vận Hành Hạ Tầng Docker (`deploy/`)

Hệ thống hỗ trợ chạy độc lập từng service lẻ để phục vụ kiểm thử hoặc chạy trọn gói cả cụm (Full Stack).

---

## 1. Danh Mục Các File Cấu Hình

| File Compose | Service Phụ Trách | Cổng Host Mặc Định | Mục Đích Kiểm Thử |
| :--- | :--- | :--- | :--- |
| `docker-compose.timescaledb.yml` | `timescaledb` (PG16 + pgvector) | `5433 -> 5432` | Kiểm thử kết nối DB, Hypertable, HNSW Vector Search, SQL Schema migration |
| `docker-compose.redis.yml` | `redis` (7-Alpine) | `6379 -> 6379` | Kiểm thử Lua Script Rate Limit, Sliding Window Queue, Context Cache |
| `docker-compose.ninerouter.yml` | `ninerouter` (Node 20) | `20128 -> 20128` | Kiểm thử Gateway xoay vòng LLM đa tài khoản |
| `docker-compose.app.yml` | `trading-bot` | Network `trading_net` | Chạy ứng dụng hoặc chạy test runner trong môi trường Docker |
| `docker-compose.yml` | **Full Stack (Cả 4 Services)** | Tất cả các cổng trên | Chạy toàn bộ hệ thống tự hành có phụ thuộc lẫn nhau |

> [!NOTE]
> Mọi container đều được tự động gắn vào mạng bridge dùng chung có tên là **`trading_net`**.
> Vì vậy, dù bạn khởi động các service bằng các lệnh compose riêng lẻ, các container vẫn có thể liên lạc với nhau qua tên container nội bộ (ví dụ: `trading_timescaledb_pgvector`, `trading_redis_governor`, `trading_9router_gateway`).

---

## 2. Hướng Dẫn Chạy Từng Service Lẻ (Standalone Testing)

### 2.1. Khởi chạy riêng TimescaleDB + pgvector
```powershell
# Khởi động ở chế độ chạy nền
docker compose -f deploy/docker-compose.timescaledb.yml up -d

# Kiểm tra logs và trạng thái healthcheck
docker compose -f deploy/docker-compose.timescaledb.yml ps
docker logs -f trading_timescaledb_pgvector

# Dừng service
docker compose -f deploy/docker-compose.timescaledb.yml down
```

### 2.2. Khởi chạy riêng Redis
```powershell
# Khởi động
docker compose -f deploy/docker-compose.redis.yml up -d

# Kiểm tra ping redis
docker exec -it trading_redis_governor redis-cli ping
# Kết quả mong đợi: PONG

# Dừng service
docker compose -f deploy/docker-compose.redis.yml down
```

### 2.3. Khởi chạy riêng 9router Gateway
```powershell
# Khởi động
docker compose -f deploy/docker-compose.ninerouter.yml up -d

# Xem logs khởi tạo
docker logs -f trading_9router_gateway

# Dừng service
docker compose -f deploy/docker-compose.ninerouter.yml down
```

### 2.4. Khởi chạy riêng Container Ứng Dụng (Trading Bot / Test Runner)
```powershell
# Build image và khởi động
docker compose -f deploy/docker-compose.app.yml up -d --build

# Chạy trực tiếp test suite hoặc verification loop bên trong container:
docker exec -it llm_trading_bot_v2_core uv run pytest tests -v
docker exec -it llm_trading_bot_v2_core bash scripts/verify_loop.sh

# Dừng container
docker compose -f deploy/docker-compose.app.yml down
```

---

## 3. Hướng Dẫn Chạy Toàn Bộ Hệ Thống (Full Stack)

```powershell
# Khởi chạy toàn bộ cả 4 services với kiểm tra healthcheck tự động
docker compose -f deploy/docker-compose.yml up -d --build

# Xem trạng thái tất cả các containers
docker compose -f deploy/docker-compose.yml ps

# Xem luồng logs tổng hợp
docker compose -f deploy/docker-compose.yml logs -f

# Dừng toàn bộ hệ thống (giữ nguyên dữ liệu volume)
docker compose -f deploy/docker-compose.yml down

# Dừng và xóa toàn bộ volumes dữ liệu (nếu muốn reset sạch sẽ)
docker compose -f deploy/docker-compose.yml down -v
```
