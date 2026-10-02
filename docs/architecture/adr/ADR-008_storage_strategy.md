# ADR-008: Data Storage & Persistence Strategy

**Status:** APPROVED  
**Date:** October 2026  
**Context:** Modern software architectures frequently introduce relational (PostgreSQL) or document (MongoDB) databases by default. In an assistive sign language translation engine, storing persistent user video or conversation history presents acute ethical and security liabilities.

## Decision
Adopt a **Minimal Local Filesystem & Zero-Database Architecture** for the operational translation platform. Raw video frames are discarded in memory, transient sliding frames are maintained in RAM deques, and persistent storage is strictly limited to model checkpoints (`.pt`), static JSON configurations, and local evaluation Parquet files.

## Evaluated Alternatives
1. **Relational Database (PostgreSQL):** Unnecessary schema overhead and database process management for an academic streaming inference application that maintains no persistent user accounts or transactional relational data.
2. **Document Database (MongoDB):** Excessive resource footprint without providing functional benefits over structured JSON/Parquet files.
3. **Cloud Object Storage (AWS S3):** Introduces remote network latency, cloud credentials management, and risks of accidental biometric data leakage.

## Consequences
- **Positive:** Zero external database setup friction; eliminates database connection pool leaks; guarantees compliance with data minimization requirements under India's DPDP Act.
- **Negative:** Session transcripts are strictly transient and are lost upon browser window closure unless explicitly exported as a text file by the user.
