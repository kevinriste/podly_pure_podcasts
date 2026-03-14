# Stage 1: Build Rust backend
FROM rust:1-bookworm AS rust-builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY migrations/ migrations/
COPY prompts/ prompts/

RUN cargo build --release

# Stage 2: Build frontend
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 3: Final runtime image
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r podly && useradd --no-log-init -r -g podly podly

WORKDIR /app
COPY --from=rust-builder /app/target/release/podly /app/podly
COPY --from=rust-builder /app/target/release/migrate_legacy /app/migrate_legacy
COPY --from=frontend-builder /app/dist /app/static
COPY migrations/ /app/migrations/
COPY prompts/ /app/prompts/

RUN mkdir -p /app/data && chown -R podly:podly /app

USER podly

ENV DATABASE_URL=sqlite:///app/data/podly.db
ENV HOST=0.0.0.0
ENV PORT=8080
ENV STATIC_DIR=/app/static
ENV DATA_DIR=/app/data

EXPOSE 8080
VOLUME ["/app/data"]

CMD ["/app/podly"]
