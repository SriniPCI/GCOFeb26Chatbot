# GCO Feb 26: The Tri-Tier Chatbot (CLI)

## Objective
Build a Console Application (Java, Kotlin, or Dart) that functions as an intelligent chatbot.

## Core Requirements
1. **Console Interface:** The app must run in a continuous terminal loop.
2. **Knowledge Base (KB):** Answer static questions using `knowledge_base.txt`.
3. **Database (DB):** Answer inventory questions by querying `inventory.db` via Function/Tool Calling.
4. **Fallback:** For any query not in the KB or DB, return: "I'm sorry, I cannot answer your query at the moment."

## Environment & Constraints
* **Database:** Use the provided `inventory.db` (SQLite) located in the root directory. If the .db file is missing or corrupted, generate a fresh one using:
```bash
sqlite3 inventory.db < inventory_setup.sql
```
* **Pathing:** Use a relative path (`./inventory.db`) for the connection string.
* **Currency:** All prices must be displayed in GBP (£).
* **Language:** Use UK English for all responses.

## Judging Criteria
* **Accuracy:** Does the bot route to the correct data source?
* **Function Calling:** Does the LLM correctly extract parameters (item name/size) for SQL?
* **Clean Code:** Adherence to best practices within the 2-hour limit.
