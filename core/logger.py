import logging
import sys
import contextvars

# Context Variables for Request & User Tracing
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

def set_request_context(request_id: str, user_id: str = None) -> None:
    request_id_ctx.set(request_id)
    if user_id:
        user_id_ctx.set(str(user_id))

def clear_request_context() -> None:
    request_id_ctx.set("-")
    user_id_ctx.set("-")

def get_user_context() -> str | None:
    val = user_id_ctx.get("-")
    return val if val != "-" else None

def get_request_context() -> str | None:
    val = request_id_ctx.get("-")
    return val if val != "-" else None

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        record.user_ctx = user_id_ctx.get("-")
        return True

def setup_logger(name: str = "EnterpriseAI") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(ContextFilter())
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(request_id)s] [%(user_ctx)s] | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
