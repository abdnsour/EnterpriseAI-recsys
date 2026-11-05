import os
import array
import time
import logging
import oracledb

from core.config import (
    uname, pwd, dsn, cdir, wltloc, wltpwd, schema,
    DB_POOL_MIN, DB_POOL_MAX, DB_POOL_TIMEOUT
)

logger = logging.getLogger(__name__)

class DBManager:
    """
    High-performance pooled connection manager for Oracle Autonomous Database (ADB 23ai).
    Implements sanitized float array bindings for native VECTOR data types.
    """
    def __init__(self):
        self.uname = uname
        self.pwd = pwd
        self.dsn = dsn
        self.cdir = cdir
        self.wltloc = wltloc
        self.wltpwd = wltpwd
        self.pool = None

    def get_connection(self):
        if not self.pool:
            logger.info(f"Initializing Oracle Database connection pool (min={DB_POOL_MIN}, max={DB_POOL_MAX})")
            pool_kwargs = {
                "user": self.uname,
                "password": self.pwd,
                "dsn": self.dsn,
                "min": DB_POOL_MIN,
                "max": DB_POOL_MAX,
                "wait_timeout": DB_POOL_TIMEOUT,
                "increment": 1
            }
            if self.cdir and os.path.isdir(self.cdir) and os.listdir(self.cdir):
                pool_kwargs["config_dir"] = self.cdir
            if self.wltloc and os.path.isdir(self.wltloc) and os.listdir(self.wltloc):
                pool_kwargs["wallet_location"] = self.wltloc
                if self.wltpwd:
                    pool_kwargs["wallet_password"] = self.wltpwd

            self.pool = oracledb.create_pool(**pool_kwargs)
        return self.pool.acquire()

    def _sanitize_bind(self, val):
        """Convert lists/tuples of floats to array.array('f') to prevent ORA-01484 in Oracle 23ai."""
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return None
            if all(isinstance(x, (int, float)) for x in val):
                # Native Oracle 23ai vector embedding
                return array.array('f', val)
            if len(val) == 1:
                return val[0]
            return str(val)
        return val

    def _sanitize_params(self, params):
        if params is None:
            return None
        if isinstance(params, dict):
            return {k: self._sanitize_bind(v) for k, v in params.items()}
        if isinstance(params, (list, tuple)):
            return [self._sanitize_bind(v) for v in params]
        return params

    def run_query(self, query: str, params=None) -> dict:
        t0 = time.time()
        logger.debug(f"[db_manager] Running query: {query.strip()[:100]}...")
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if schema:
                cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema}")
            if params is not None:
                cursor.execute(query, self._sanitize_params(params))
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            duration_ms = (time.time() - t0) * 1000
            logger.info(f"[db_manager] Query completed in {duration_ms:.1f}ms | rows={len(result)}")
            return {"is_data_found": True, "data": result}
        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            logger.error(f"[db_manager] Query error after {duration_ms:.1f}ms: {e}", exc_info=True)
            return {"is_data_found": False, "data": f"Error running query: {str(e)}"}
        finally:
            conn.close()

    def run_query_with_columns(self, query: str, params=None) -> dict:
        t0 = time.time()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if schema:
                cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema}")
            if params is not None:
                cursor.execute(query, self._sanitize_params(params))
            else:
                cursor.execute(query)
            columns = [col[0] for col in cursor.description] if cursor.description else []
            result = cursor.fetchall()
            formatted_data = []
            for row in result:
                formatted_row = []
                for item in row:
                    if isinstance(item, oracledb.LOB):
                        formatted_row.append(item.read())
                    else:
                        formatted_row.append(item)
                formatted_data.append(dict(zip(columns, formatted_row)))
            duration_ms = (time.time() - t0) * 1000
            return {"is_data_found": True, "columns": columns, "data": formatted_data}
        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            logger.error(f"[db_manager] Query with columns error after {duration_ms:.1f}ms: {e}", exc_info=True)
            return {"is_data_found": False, "columns": [], "data": f"Error: {str(e)}"}
        finally:
            conn.close()

db_manager = DBManager()
