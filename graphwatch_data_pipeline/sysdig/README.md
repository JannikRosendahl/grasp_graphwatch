# Sysdig Data Pipeline

This directory contains the Sysdig data pipeline for Graphwatch.

## Requirements

- `grasp` installed
- PostgreSQL set up and running
- A configured `.env` file
- Input data placed in `input/sysdig_scaps`

## Setup

1. Follow the PostgreSQL setup instructions in `../db_setup/README.md`.
2. Create and configure the `.env` file with the correct PostgreSQL connection details.
```bash
cp .env_example .env
```
3. Ensure the input directory exists and place the data in `input/sysdig_scaps`.
4. Make all files executable:

```bash
chmod +x ./*.py
```

## Run

After placing the input data in `input/sysdig_scaps`, run:

```bash
source .env
./main.py
```

## Notes

- Make sure `grasp` is installed before running.
- Verify your `.env` settings match your PostgreSQL configuration.
