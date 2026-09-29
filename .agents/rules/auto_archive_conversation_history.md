# Rule: Automatic Local Conversation Transcript & History Sync

1. **Automatic Archival Requirement:**
   - Whenever completing a task, answering user questions, or making system changes, the agent must automatically ensure the latest user prompts, responses, decisions, and transcripts are synchronized into `conversation_history/`.
   - Run `python conversation_history/sync_conversation_history.py` to auto-extract and preserve untruncated transcripts from the IDE brain folder.

2. **Git Privacy Enforcement:**
   - The `conversation_history/` directory MUST always remain in `.gitignore` to prevent committing or pushing local transcripts and sync scripts to the remote repository.
