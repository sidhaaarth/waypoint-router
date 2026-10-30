# Run Waypoint in your own network

This folder has everything needed to build and run Waypoint on your own machine or server: sign-up and login, onboarding,
the dashboard, live chat, the docs and the router API. It contains no demo data. You bring your own rules and conversations.

## What you need

- Docker with Docker Compose (Docker Desktop on Windows or Mac).
- A Nebius Token Factory API key. You paste it at sign-up, and it is stored encrypted in your own volume.

## Start it

```bash
docker compose up --build -d
```

Open http://localhost:8888/signup, enter your details and Nebius Token Factory key, and save the router key you get back.
Log in with it at http://localhost:8888/login, then set up your first connection.

## Where your data lives

Projects, keys, request history, uploads and memory of past cases are stored in the Docker volume `waypoint-data`.
They survive restarts and upgrades. To back up, copy the volume. To wipe everything: `docker compose down -v`.

## Optional settings

Create a `.env` file next to `docker-compose.yml`:

```
TAVILY_API_KEY=        # look-ups on your trusted help-centre sites
LANGSMITH_API_KEY=     # tracing of every model call
LANGSMITH_PROJECT=waypoint
ROUTER_SECRET=         # your own encryption key (Fernet); generated in the volume if unset
```

## Good to know

- Waypoint and all your data stay in your network. The model calls go to Nebius Token Factory over HTTPS, because
  that is where NVIDIA Nemotron runs.
- To change the port, edit the left side of `"8888:8888"` in `docker-compose.yml`.
- Health check: http://localhost:8888/healthz
- Full docs: http://localhost:8888/docs
