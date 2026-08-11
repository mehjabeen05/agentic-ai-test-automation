# data/

This directory holds the local SQLite database used for persistence
(`test_automation.db`, see Step 10 in the project README).

The `.db` file itself is gitignored — it's local, generated data, not
source. Only this README is tracked, so the directory exists on a fresh
checkout; `core/database.py` creates the database file and its tables
automatically the first time anything connects.
