import logging
import sys
import json
import datetime
import traceback
import fcntl
from pathlib import Path

def setup_logging(verbose: int = 0) -> logging.Logger:
    """
    Configures the root logger.
    
    - INFO level goes to stdout (standard application output).
    - ERROR/WARNING goes to stderr.
    - DEBUG level goes to stderr if verbose >= 2.
    """
    logger = logging.getLogger("arc_agi")
    logger.setLevel(logging.DEBUG if verbose >= 2 else logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Handler for standard info/results (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Simple format for CLI output, or just message for cleanliness if we are printing tables manually
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Handler for debug/errors (stderr)
    # We filter this so it only takes ERROR (always) and DEBUG (if verbose)
    # But standard StreamHandler(sys.stderr) takes everything >= level.
    
    # If verbose, we want detailed logs to stderr
    if verbose >= 2:
        debug_handler = logging.StreamHandler(sys.stderr)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)
    
    # We generally don't attach the console_handler to the root logger if we are 
    # manually printing tables with print(). 
    # However, for this refactor, we will use logger.info() for standard messages.
    # But carefully: print_table_header/row in reporting.py uses print() for strict formatting.
    # We should probably leave "presentation" prints as prints, and usage logs as logs.
    
    return logger

class PrefixedStdout:
    def __init__(self, prefix, message_width=None):
        self.prefix = prefix
        self.message_width = message_width
        self.original_stdout = sys.stdout
        self.at_line_start = True

    def write(self, text):
        if not text:
            return
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if i > 0:
                # We encountered a newline in the split
                self.original_stdout.write('\n')
                self.at_line_start = True
            
            if line:
                if self.at_line_start:
                    if callable(self.prefix):
                        p = self.prefix()
                    else:
                        p = self.prefix
                    self.original_stdout.write(p)
                    self.at_line_start = False
                
                if self.message_width and len(line) > self.message_width:
                    line = line[:self.message_width - 3] + "..."
                self.original_stdout.write(line)
        
    def flush(self):
        self.original_stdout.flush()
        
    def write_raw(self, text):
        """
        Writes to the underlying stream without any prefixing.
        Used by StderrToStdoutRedirector to bypass the table formatting.
        """
        if not text:
            return
        
        self.original_stdout.write(text)
        
        # Update state tracking based on the last character written
        # If the last character was a newline, we are at the start of a new line.
        if text.endswith('\n'):
            self.at_line_start = True
        else:
            self.at_line_start = False
        
    def __enter__(self):
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout

class StderrToStdoutRedirector:
    """
    Redirects stderr writes to stdout.
    If the current stdout is a PrefixedStdout (i.e., we are drawing a table),
    it uses 'write_raw' to bypass the table prefix, ensuring error messages
    appear as full-width, unformatted text.
    """
    def write(self, text):
        # Always resolve sys.stdout dynamically because it changes 
        # when entering/exiting PrefixedStdout contexts.
        current_stdout = sys.stdout
        
        if hasattr(current_stdout, "write_raw"):
            current_stdout.write_raw(text)
        else:
            current_stdout.write(text)

    def flush(self):
        sys.stdout.flush()

    def reconfigure(self, *args, **kwargs):
        # Proxy to stdout if possible
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(*args, **kwargs)

_CURRENT_LOG_DIR = "logs"

def set_log_dir(path: str):
    global _CURRENT_LOG_DIR
    _CURRENT_LOG_DIR = path

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"arc_agi.{name}")

def log_failure(
    run_timestamp: str,
    task_id: str,
    run_id: str,
    error: Exception,
    model: str = None,
    step: str = None,
    test_index: int = None,
    is_retryable: bool = False,
    log_dir: str = None
):
    """
    Logs a structured failure record to a JSONL file.
    Designed to be a 'System of Record' for terminal failures.
    """
    try:
        # Construct the failures file path: {log_dir}/{timestamp}_failures.jsonl
        ldir = log_dir if log_dir is not None else _CURRENT_LOG_DIR
        log_path = Path(ldir)
        log_path.mkdir(exist_ok=True, parents=True)
        failures_path = log_path / f"{run_timestamp}_failures.jsonl"

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_id": task_id,
            "test_index": test_index,
            "step": step,
            "model": model,
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc() if sys.exc_info()[0] else "None (Logical Error)",
            "is_retryable": is_retryable
        }

        # Serialize safely (handle non-serializable types)
        json_entry = json.dumps(entry, default=str)

        with open(failures_path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json_entry + "\n")
                f.flush() # Ensure it hits the disk
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                
    except Exception as e:
        # Fallback: Don't let logging crash the application
        print(f"CRITICAL: Failed to log failure to DLQ: {e}", file=sys.stderr)

def write_step_log(step_name: str, data: dict, timestamp: str, task_id: str, test_index: int, verbose: bool = False, log_dir: str = None):
    ldir = log_dir if log_dir is not None else _CURRENT_LOG_DIR
    log_path = Path(ldir) / f"{timestamp}_{task_id}_{test_index}_{step_name}.json"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=4, default=lambda o: '<not serializable>')
    if verbose:
        print(f"Saved log for {step_name} to {log_path}")
