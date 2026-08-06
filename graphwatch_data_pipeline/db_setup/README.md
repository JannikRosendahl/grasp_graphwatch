# Database Setup

This directory contains the Docker Compose setup for the PostgreSQL database used by the Graphwatch data pipeline.

## Prerequisites

- Docker and Docker Compose installed

## Setup

1. Create your `.env` file from the example:

    ```bash
    cp .env_example .env
    ```

2. Adjust the values in `.env` if needed:
    - `POSTGRES_USER` — database user
    - `POSTGRES_PASSWORD` — database password
    - `POSTGRES_PORT` — host port the database is exposed on
    - `POSTGRES_DB_NAME` — database name

3. Start the database:

    ```bash
    docker compose --env-file .env up -d
    ```

## Verify

Check that the container is running and healthy:

```bash
docker-compose ps
```

Tail the logs:

```bash
docker-compose logs -f grasp_graphwatch_postgres
```

## Stopping

Stop the database:

```bash
docker-compose down
```

## Notes

- `docker-compose.yml` bind-mounts `./.postgres_data_grasp_graphwatch` into the container as the Postgres data directory, so data survives `docker-compose down`/`up`. `docker-compose down -v` will **not** remove it (there's no named volume, only a bind mount) — to fully reset the database, stop the container and delete that directory:

    ```bash
    docker-compose down
    rm -rf ./.postgres_data_grasp_graphwatch
    ```

- The values from `.env` must match the connection settings used by the data pipeline scripts (e.g. `db_host`, `db_port`, `db_user`, `db_password`, `db_name` in `graphwatch_data_pipeline/sysdig/.env_upload`).
