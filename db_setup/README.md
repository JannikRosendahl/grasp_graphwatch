# DB Setup

- You can follow the complete installation guide from [PIDSMaker](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/), or use this shortcut.  
- This installation is tailored for the Postgres DB using the .dump files from PIDSMaker. 
- We base ourselves on the installation guide from [PIDSMaker](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/).

## Ensure Docker is installed 
0) [Docker](https://docs.docker.com/engine/install/)

## Download Datasets

1) Download Postgres dump Files
Download dataset dumps from the [PIDSMaker Google Drive](https://drive.google.com/drive/folders/1hqfz8__zVqb3QzBuOI2SxrW4lLIdYqFr) or via the [PIDSMaker docs](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/). 
- Get the share links (right-click → Share → Get link) for:
- `cadets_e5.dump` (timestamp: 05.06.2025)
- `theia_clearscope_e5.tar` (timestamp: 06.06.2025)
- `optc_and_cadets_theia_clearscope_e3.tar` (timestamp: 13.09.2025)
- Update [.env_example](./.env_example) with your links, save it as `.env` in `db_setup`, then run `download_files.py` to fetch the files.

```bash
python download_files.py
```
- Uncompress the .tar files
```bash
tar -xvf optc_and_cadets_theia_clearscope_e3.tar
tar -xvf theia_clearscope_e5.tar
```


2) Clone and enter the repo:

```bash
git clone https://github.com/ubc-provenance/PIDSMaker
cd PIDSMaker
git checkout af14d16caeca6727b36a50b4b7333c472febe764
```

3) Create and fill data directory


```bash
mkdir -p data
mv ../*.dump data/
``` 

4) Prepare .env.local

- If no service is running on port 8888 and all steps in this README are followed, the defaults will work.

5) Run docker compose

```bash
docker compose -p postgres_pidsmaker -f compose-postgres.yml --env-file .env.local up -d 
```

6) Load dumps:

```bash
docker compose -p postgres_pidsmaker exec postgres bash
pg_restore -U postgres -h localhost -p 5432 -d cadets_e3 /data/cadets_e3.dump
pg_restore -U postgres -h localhost -p 5432 -d theia_e3 /data/theia_e3.dump
pg_restore -U postgres -h localhost -p 5432 -d clearscope_e3 /data/clearscope_e3.dump
pg_restore -U postgres -h localhost -p 5432 -d optc051 /data/optc051.dump
...
```

```
Repeat the `pg_restore` command for each of the other `.dump` files in the `data/` directory.
```
