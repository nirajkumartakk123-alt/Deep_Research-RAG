# Deployment Guide

## Local (Docker Compose) — recommended for demo/portfolio use

1. `cp .env.example .env` and fill in `GROQ_API_KEY` and `TAVILY_API_KEY`.
2. `docker compose up --build`
3. Apply the DB migration once containers are healthy: