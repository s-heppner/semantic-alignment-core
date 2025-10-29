# smr_discovery 

A Service for discovering Semantic Match Registry (SMR) services for given `semantic_ids`

## How to Use
There are two main options to run:

### Run `service.py`
You need a working Python installation (3.11 or higher).
- Clone this repository
- Create a new virtual environment: `python3 -m venv venv` (Note that the service is only tested on Linux systems)
- Activate the virtual environment: `source venv/bin/activate`
- Install the package: `pip install .`
- Copy the `config.ini.default` to `config.ini` and adapt the fields as necessary
- Run the service: 

```commandline
python -m smr_discovery.service \
  --endpoints-json-file ./configs/endpoints.json \
  --endpoint http://localhost:8125 \
  --listen-address 127.0.0.1 \
  --port 8125

```

### Run via Docker
To run via docker, you obviously need a working docker installation.
- Clone the repository
- Copy the `config.ini.default` to `config.ini` and adapt the fields as necessary
- Choose one of the two options below:

**Option 1:** Build and run manually

In the project root directory:
```commandline
docker build -t smr_discovery .
```
```commandline
docker run --rm -p 8125:8125 smr_discovery
```

> [!note]
> You may have to change the port in the run command if you changed it in your `config.ini`. 

**Option 2:** Use the docker compose file

In the project root directory:
```commandline
docker compose up
```

> [!note]
> If you changed your `config.ini`, you might need to adapt the respective fields in the `compose.yaml`. 

