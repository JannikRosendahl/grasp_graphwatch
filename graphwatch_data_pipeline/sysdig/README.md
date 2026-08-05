# Sysdig Data Pipeline

This directory contains the Sysdig data pipeline for Graphwatch.

## Requirements

- `grasp` installed
- PostgreSQL set up and running
- A configured `.env` file
- Input data placed in `input/sysdig_scaps`

## Setup

1. Set up PostgreSQL.
2. Create and configure the `.env` file.
3. Make all files executable:

	```bash
	chmod +x ./*.py
	```

## Run

After placing the input data in `input/sysdig_scaps`, run:

```bash
source .env_upload
./main.py
```

## Notes

- Make sure `grasp` is installed before running.
- Verify your `.env` settings match your PostgreSQL configuration.
