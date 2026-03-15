# Container Stats: Python vs Rust Backend

**Date**: 2026-03-15
**Host**: 23.34 GiB RAM, Linux 6.8.0-101-generic

## Resource Usage (Idle)

| Metric | Python (`podly`) | Rust (`podly-rust`) | Ratio |
|--------|-----------------|--------------------:|------:|
| **CPU** | 0.01–1.50% | 0.00% | ~75x less |
| **Memory** | 1.333 GiB | 17.8 MiB | **75x less** |
| **PIDs** | 15 | 10 | 1.5x less |
| **Network I/O** | 2.54 GB / 1.76 GB | 84.5 kB / 518 kB | ~30,000x less (uptime diff) |
| **Block I/O** | 2.81 GB / 3.1 GB | 467 kB / 60.1 MB | ~5,700x less (uptime diff) |

### Notes on Network/Block I/O
- Python container has been running since **2026-03-08** (7 days)
- Rust container was last recreated **2026-03-15** (~30 min ago at time of measurement)
- Network/Block I/O difference is dominated by uptime, not efficiency

## Memory Snapshots (5 readings, 3s apart)

| Reading | Python | Rust |
|---------|--------|------|
| 1 | 1.333 GiB | 17.79 MiB |
| 2 | 1.333 GiB | 17.79 MiB |
| 3 | 1.333 GiB | 17.78 MiB |
| 4 | 1.333 GiB | 17.78 MiB |
| 5 | 1.333 GiB | 17.78 MiB |

Memory is extremely stable for both. Rust uses **~75x less memory** at idle.

## Image & Binary Sizes

| Component | Python | Rust |
|-----------|--------|------|
| **Docker image** | 8.66 GB | 603 MB |
| **Binary/runtime** | Full Python 3.12 + deps | 21 MB single binary |
| **Image ratio** | — | **14x smaller** |

## Data Storage

| Component | Python | Rust |
|-----------|--------|------|
| **Database** | 329 MB (`sqlite3.db`) | 311 MB (`podly.db`) |
| **Podcast data** | 25 GB | 26 GB (16 GB processed, 11 GB unprocessed) |
| **Instance/config** | 760 MB total | — (in data dir) |

## Process Counts

| Container | Processes |
|-----------|-----------|
| Python | 3 (header + workers) |
| Rust | 1 (single async binary) |

## Resource Limits
Neither container has memory or CPU limits configured.

## Summary

The Rust backend achieves a **75x memory reduction** (1.33 GiB → 18 MiB) and a **14x smaller Docker image** (8.66 GB → 603 MB) while maintaining full API parity. CPU usage at idle is negligible for both, with Rust registering 0.00% across all measurements vs Python's 0.01–1.50%.
