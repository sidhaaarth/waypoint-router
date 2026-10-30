# Waypoint

Nemotron-first routing for customer support, powered by NVIDIA Nemotron on Nebius Token Factory.

Waypoint sits between your support app and Token Factory. Point your OpenAI client's `base_url` at it and every email or
chat goes through checks, rules and store look-ups before the cheapest Nemotron model that can do the job writes the
reply. Nano answers most messages; Super and Ultra step in only when needed.

## What it does

- **Cascade routing**: Nano first, with logprob confidence; Super or Ultra only when Nano is unsure.
- **Support pipeline**, per message:
  turn check → photo check (Nemotron-3.5-Super-VL) → scope and safety gates → understand (Nano + memory of past cases)
  → extract (exact match + Nano + store record) → store rules (plain code) → trusted-site lookup (Tavily) → store actions
  → writer (Nano / Super / Ultra) → safety check → decision record.
- **Memory of past cases**: solved conversations are embedded once (Qwen3-Embedding on Token Factory); the 3 nearest go into
  Nano's prompt. No training, no GPU, and it grows nightly from a store's own good conversations (opt-in).
- **Gateway features**: rate-limit pacing and fallback, cost per customer, prompt-cache hints, LangSmith tracing.
- **Web app**: product site, sign-up with a "teach your router" step, onboarding, dashboard with analytics, live chat, and
  a docs page.

## Run it locally

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env     # optional: Tavily, LangSmith
.venv\Scripts\python -m uvicorn app.main:app --port 8888
```

Open http://localhost:8888/signup, sign up with your Nebius Token Factory key, and log in with the router key you get back.

## Use it from your app

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8888/v1", api_key="rk-...")   # your Waypoint router key
reply = client.chat.completions.create(
    model="support",            # your connection name
    user="customer-42",         # optional: cost per customer
    messages=[{"role": "user", "content": "My boots arrived torn, order 4223520702"}],
)
```

## Run it in your own network (Docker)

The Docker image is the plain app: sign-up, login, onboarding, dashboard, live chat, docs and the router API. It has no
demo store, rulebook, memory index or showcase pages. Users download it from `/docs#selfhost` (`/download/waypoint.zip`)
or build from this repo:

```bash
docker compose up --build -d     # then open http://localhost:8888/signup
```

Data lives in the `waypoint-data` volume. See [SELF_HOST.md](SELF_HOST.md), which ships in the zip as its README.

## Project layout

| Path | What |
|---|---|
| `app/` | Router: API (`main.py`), cascade, pipeline, rules engine, memory, vision, web lookup, rate limits, costs, tracing |
| `static/` | Site, sign-up and onboarding, dashboard, live chat, docs, brand assets |
| `Dockerfile`, `docker-compose.yml`, `SELF_HOST.md` | Self-hosted image and its instructions |

The demo store, datasets, tests, benchmarks and planning notes are kept out of the repo.

## Data and licences

- ABCD (Action-Based Conversations Dataset, ASAPP): MIT.
- Benchmark sets used during development: Bitext (CDLA-Sharing), Banking77 (CC BY 4.0), support tickets (CC BY-NC, testing
  only). Review photos used in testing are not included (no clear licence).
