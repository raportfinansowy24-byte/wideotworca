import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def init_db(db_path):
    """Inicjalizacja tabeli historii renderów"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS renders (
        job_id TEXT PRIMARY KEY,
        topic TEXT,
        status TEXT,
        video_url TEXT,
        error TEXT,
        video_duration REAL,
        created_at TIMESTAMP,
        completed_at TIMESTAMP,
        current_stage TEXT,
        checkpoint_data TEXT,
        paused_at TIMESTAMP,
        paused_reason TEXT,
        retry_count INTEGER DEFAULT 0,
        next_retry_at TIMESTAMP
    )''')
    # Migrate existing databases that are missing the checkpoint columns
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(renders)")}
    for col, col_def in [
        ("current_stage",   "TEXT"),
        ("checkpoint_data", "TEXT"),
        ("paused_at",       "TIMESTAMP"),
        ("paused_reason",   "TEXT"),
    ]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE renders ADD COLUMN {col} {col_def}")
    conn.commit()
    conn.close()

def ensure_retry_columns(db_path):
    """Add retry_count and next_retry_at columns if they don't exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Check if columns exist
    c.execute("PRAGMA table_info(renders)")
    columns = {row[1] for row in c.fetchall()}

    if "retry_count" not in columns:
        c.execute("ALTER TABLE renders ADD COLUMN retry_count INTEGER DEFAULT 0")
        logger.info("✅ Added retry_count column to renders table")

    if "next_retry_at" not in columns:
        c.execute("ALTER TABLE renders ADD COLUMN next_retry_at TIMESTAMP")
        logger.info("✅ Added next_retry_at column to renders table")

    conn.commit()
    conn.close()

def save_render_to_db(db_path, job_id, topic, status, video_url=None, error=None, video_duration=None):
    """Zapis renderowania do bazy danych"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    completed_at = datetime.utcnow() if status in ['success', 'failed'] else None
    c.execute('''INSERT OR REPLACE INTO renders
                 (job_id, topic, status, video_url, error, video_duration, created_at, completed_at,
                  current_stage, checkpoint_data, paused_at, paused_reason)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                  COALESCE((SELECT current_stage  FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT checkpoint_data FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT paused_at       FROM renders WHERE job_id = ?), NULL),
                  COALESCE((SELECT paused_reason   FROM renders WHERE job_id = ?), NULL))''',
              (job_id, topic, status, video_url, error, video_duration, datetime.utcnow(), completed_at,
               job_id, job_id, job_id, job_id))
    conn.commit()
    conn.close()

def save_checkpoint(db_path, job_id, stage, data=None, error=None, auto_retry_enabled=False, max_attempts=3, initial_delay=30, max_delay=600):
    """Save a checkpoint so the job can be resumed from this stage on error."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    checkpoint_json = json.dumps(data or {})

    if error is not None:
        # Get current retry count
        c.execute("SELECT retry_count FROM renders WHERE job_id = ?", (job_id,))
        row = c.fetchone()
        current_retry_count = (row[0] if row and row[0] is not None else 0) + 1

        # Calculate next retry time with exponential backoff
        if auto_retry_enabled and current_retry_count <= max_attempts:
            delay = min(
                initial_delay * (2 ** (current_retry_count - 1)),
                max_delay
            )
            next_retry_time = datetime.utcnow() + timedelta(seconds=delay)
            logger.info(f"⏰ Job {job_id} will auto-retry in {delay}s (attempt {current_retry_count}/{max_attempts})")
        else:
            next_retry_time = None
            logger.warning(f"❌ Job {job_id} exceeded max retry attempts ({max_attempts})")

        # Job hit an error – pause it
        c.execute('''UPDATE renders
                     SET current_stage   = ?,
                         checkpoint_data = ?,
                         paused_at       = ?,
                         paused_reason   = ?,
                         status          = 'paused',
                         retry_count     = ?,
                         next_retry_at   = ?
                     WHERE job_id = ?''',
                  (stage, checkpoint_json, datetime.utcnow(), error, current_retry_count, next_retry_time, job_id))
    else:
        # Successful stage – save progress
        c.execute('''UPDATE renders
                     SET current_stage   = ?,
                         checkpoint_data = ?,
                         retry_count     = 0,
                         next_retry_at   = NULL
                     WHERE job_id = ?''',
                  (stage, checkpoint_json, job_id))

    conn.commit()
    conn.close()
    logger.info(f"💾 Checkpoint saved: job={job_id} stage={stage}")

def get_checkpoint(db_path, job_id):
    """Return checkpoint info for a paused job, or None if not found / not paused."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''SELECT current_stage, checkpoint_data, paused_at, paused_reason, status, topic,
                        retry_count, next_retry_at
                 FROM renders WHERE job_id = ?''', (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "current_stage":   row[0],
        "checkpoint_data": json.loads(row[1]) if row[1] else {},
        "paused_at":       row[2],
        "paused_reason":   row[3],
        "status":          row[4],
        "topic":           row[5],
        "retry_count":     row[6] if row[6] is not None else 0,
        "next_retry_at":   row[7],
    }

def get_render_from_db(db_path, job_id):
    """Pobranie statusu renderowania z bazy"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''SELECT job_id, topic, status, video_url, error, video_duration,
                        created_at, completed_at, current_stage, checkpoint_data,
                        paused_at, paused_reason, retry_count, next_retry_at
                 FROM renders WHERE job_id = ?''', (job_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            "job_id":          row[0],
            "topic":           row[1],
            "status":          row[2],
            "video_url":       row[3],
            "error":           row[4],
            "video_duration":  row[5],
            "created_at":      row[6],
            "completed_at":    row[7],
            "current_stage":   row[8],
            "checkpoint_data": json.loads(row[9]) if row[9] else {},
            "paused_at":       row[10],
            "paused_reason":   row[11],
            "retry_count":     row[12] if row[12] is not None else 0,
            "next_retry_at":   row[13],
        }
    return None
