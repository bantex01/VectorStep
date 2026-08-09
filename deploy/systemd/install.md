# Installing VectorStep on a VM (systemd)

FHS-ish layout, identical in shape to the Gateway's
(`../../../VectorStep-Gateway/deploy/systemd/install.md`):

| What | Where |
|---|---|
| Code (checkout + venv) | `/opt/vectorstep/vectorstep/` |
| Config | `/etc/vectorstep/vectorstep/` |
| State (db, artifacts, pipelines, steps) | `/var/lib/vectorstep/vectorstep/` |
| Logs | `/var/log/vectorstep/vectorstep/` |
| Secrets | `/etc/vectorstep/vectorstep/env` (root:vectorstep, mode 640) |

## Install

1. Create the system user:
   ```sh
   sudo useradd -r -s /usr/sbin/nologin -d /opt/vectorstep/vectorstep vectorstep
   ```

2. Clone the repo and create the venv:
   ```sh
   sudo mkdir -p /opt/vectorstep
   sudo git clone https://github.com/bantex01/VectorStep.git /opt/vectorstep/vectorstep
   cd /opt/vectorstep/vectorstep
   sudo python3 -m venv .venv
   sudo .venv/bin/pip install -r requirements.txt
   ```

3. Create the config/state/log trees with ownership:
   ```sh
   sudo mkdir -p /etc/vectorstep/vectorstep
   sudo mkdir -p /var/lib/vectorstep/vectorstep/{db,artifacts,pipelines,steps}
   sudo mkdir -p /var/log/vectorstep/vectorstep
   sudo chown -R vectorstep:vectorstep /opt/vectorstep/vectorstep \
     /var/lib/vectorstep/vectorstep /var/log/vectorstep/vectorstep
   sudo chown -R root:vectorstep /etc/vectorstep/vectorstep
   ```

4. Copy the sample config and point its writable paths at `/var/lib` and `/var/log`:
   ```sh
   sudo cp samples/config.yaml.example /etc/vectorstep/vectorstep/config.yaml
   sudo $EDITOR /etc/vectorstep/vectorstep/config.yaml
   ```
   Set at minimum:
   ```yaml
   database:
     url: sqlite+aiosqlite:////var/lib/vectorstep/vectorstep/db/runs.db
   pipeline_config_dir: /var/lib/vectorstep/vectorstep/pipelines
   step_library_dir: /var/lib/vectorstep/vectorstep/steps
   artifacts:
     dir: /var/lib/vectorstep/vectorstep/artifacts
   logging:
     dir: /var/log/vectorstep/vectorstep
   executors:
     gateway:
       url: ws://<gateway-host>:18780/rpc
       token: ${VECTORSTEP_GATEWAY_TOKEN}
       rest_url: http://<gateway-host>:18780
   ```
   (See `../../samples/config.yaml.example` for the full annotated reference —
   auth, notifications, human-approval routing, pricing, replay, etc.)

5. Create the env file with your secrets:
   ```sh
   sudo cp env.example /etc/vectorstep/vectorstep/env
   sudo $EDITOR /etc/vectorstep/vectorstep/env
   sudo chown root:vectorstep /etc/vectorstep/vectorstep/env
   sudo chmod 640 /etc/vectorstep/vectorstep/env
   ```

6. Install and start the unit:
   ```sh
   sudo cp deploy/systemd/vectorstep.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now vectorstep
   sudo systemctl status vectorstep
   ```

7. Verify:
   ```sh
   curl -s http://localhost:8000/health
   ```

## Upgrade

```sh
cd /opt/vectorstep/vectorstep
sudo -u vectorstep git pull
sudo -u vectorstep .venv/bin/pip install -r requirements.txt
sudo systemctl restart vectorstep
```

`restart` is required (not `reload`) for anything that touches code or
dependencies — it's also what runs any pending Alembic migration
(`database.auto_migrate: true`, the default, runs it automatically on boot;
see `../../samples/config.yaml.example`'s database comment for the
`auto_migrate: false` alternative).

For a **YAML-only** change (editing a pipeline or step file, or
`config.yaml` itself), `reload` is enough and avoids dropping in-flight runs:

```sh
sudo systemctl reload vectorstep
journalctl -u vectorstep -n 50   # confirm the reload log line
```
