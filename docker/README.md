# CVAT Docker Setup

Simple CVAT (Computer Vision Annotation Tool) setup for thermal image annotation.

## Quick Start

```bash
# Start CVAT
cd docker
docker compose up -d

# View logs
docker compose logs -f

# Stop CVAT
docker compose down
```

## Access

- **CVAT UI**: http://localhost:1280
- **CVAT API**: http://localhost:12808

## First Time Setup

1. Start the containers:
   ```bash
   docker compose up -d
   ```

2. Wait for initialization (check logs):
   ```bash
   docker compose logs -f cvat
   ```

3. Create superuser account:
   ```bash
   docker compose exec cvat python3 ~/manage.py createsuperuser
   ```

4. Access CVAT at http://localhost:1280 and login

## Data Persistence

All data is stored in `./volumes/` (gitignored):
- `cvat_db/` - PostgreSQL database
- `cvat_data/` - Annotation data and uploaded images
- `cvat_keys/` - Authentication keys
- `cvat_logs/` - Application logs
- `cvat_models/` - ML models (if used)

## Container Architecture

- **cvat_db**: PostgreSQL 15 database
- **cvat_redis**: Redis cache
- **cvat**: Main CVAT server
- **cvat_ui**: Web UI (Nginx)
- **cvat_worker_import**: Import task worker
- **cvat_worker_export**: Export task worker
- **cvat_worker_annotation**: Annotation processing worker
- **cvat_worker_webhooks**: Webhook worker
- **opa**: Open Policy Agent for authorization

## Useful Commands

```bash
# View all containers
docker compose ps

# Restart CVAT
docker compose restart cvat

# View database logs
docker compose logs cvat_db

# Backup database
docker compose exec cvat_db pg_dump -U root cvat > backup.sql

# Clean everything (WARNING: deletes all data)
docker compose down -v
rm -rf volumes/*
```

## Notes

- Uses CVAT v2.11.0 (stable release)
- No auto-annotation/inference server configured yet
- All volumes are gitignored for security
- Redis and PostgreSQL run in network-isolated mode
