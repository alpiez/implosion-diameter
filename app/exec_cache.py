"""Phase 6c: caches for tests/test_executor.py, built from the PINNED commit at image build.

The executor tests spend about half of all verification CPU on two things that do not
depend on the patch:
  * setUpClass parses every TPC-H / TPC-DS CSV into Python rows (about 1.4 s per job);
  * cached_execute / test_optimized_tpch run each query through DuckDB for the expected
    result (about a quarter of test_execute_tpcds).
build() (Dockerfile) captures both by running the ORIGINAL pinned test code:
  rows  {(dataset, table): rows}   what setUpClass passes to Table(columns=..., rows=...)
  duck  {duckdb sql: DataFrame}    DuckDB's result for that exact SQL string
At runtime (verifier child, only when the patch leaves the test file, tests/helpers.py and
the TPC data untouched), install() makes the test module
  * build its tables from the cached rows (Table is still the patched class), and
  * get DuckDB results through a connection wrapper: a query whose SQL string, as the
    PATCHED sqlglot transpiles it, equals a cached one returns a copy of the cached frame;
    anything else goes to DuckDB. DuckDB's result is a function of the SQL string and the
    data only, so this is exact.
"""

import pickle
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache" / "executor.pkl"
DATASETS = (("tpch", "DIR_TPCH", "TPCH_SCHEMA", "tpch_conn", "tpch_tables"),
            ("tpcds", "DIR_TPCDS", "TPCDS_SCHEMA", "tpcds_conn", "tpcds_tables"))
# a patch touching any of these disables the caches for that commit
GUARDED = ("tests/test_executor.py", "tests/helpers.py",
           "tests/fixtures/optimizer/tpc-h/", "tests/fixtures/optimizer/tpc-ds/")

DATA: dict | None = None  # loaded once in the fork server, inherited by the children


def guard_ok(paths) -> bool:
    return not any(p and p.startswith(GUARDED) for p in paths)


def _path() -> Path:
    return CACHE if CACHE.exists() else Path("/app/cache/executor.pkl")  # tools run from /w


def available() -> bool:
    """Cheap check (no unpickling) used when planning."""
    try:
        return _path().stat().st_size > 0
    except OSError:
        return False


def load() -> bool:
    global DATA
    if DATA is None:
        try:
            with _path().open("rb") as f:
                DATA = pickle.load(f)
        except Exception:  # noqa: BLE001 - no cache: the tests run as usual
            DATA = {}
    return bool(DATA)


class _Fetched:
    """Result of a cached query: fetchdf() returns a copy; anything else runs the query."""

    def __init__(self, conn, sql, df):
        self._conn, self._sql, self._df, self._real = conn, sql, df, None

    def fetchdf(self):
        return self._df.copy(deep=True)

    def __getattr__(self, name):
        if self._real is None:
            self._real = self._conn.execute(self._sql)
        return getattr(self._real, name)


class _Conn:
    def __init__(self, conn, duck):
        self._c, self._duck = conn, duck

    def execute(self, sql, *a, **k):
        if not a and not k and isinstance(sql, str):
            df = self._duck.get(sql)
            if df is not None:
                return _Fetched(self._c, sql, df)
        return self._c.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._c, name)


class _DuckDB:
    def __init__(self, real, duck):
        self._real, self._duck = real, duck

    def connect(self, *a, **k):
        return _Conn(self._real.connect(*a, **k), self._duck)

    def __getattr__(self, name):
        return getattr(self._real, name)


def install(mod) -> bool:
    """Use the caches in the (freshly imported, patched-tree) tests.test_executor module."""
    if not DATA or not hasattr(mod, "TestExecutor"):
        return False
    rows, duck = DATA["rows"], DATA["duck"]
    mod.duckdb = _DuckDB(mod.duckdb, duck)

    def setUpClass(cls):
        # pinned setUpClass, with the CSV parsing replaced by the rows it produced
        cls.tpch_conn = mod.duckdb.connect()
        cls.tpcds_conn = mod.duckdb.connect()
        cls.tpch_tables = {}
        cls.tpcds_tables = {}
        for name, dir_attr, schema_attr, conn_attr, tables_attr in DATASETS:
            conn, tables = getattr(cls, conn_attr), getattr(cls, tables_attr)
            for table, columns in getattr(mod, schema_attr).items():
                file_name = f"{getattr(mod, dir_attr)}{table}.csv.gz"
                conn.execute(
                    f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
                )
                tables[table] = mod.Table(columns=columns, rows=rows[(name, table)])
        cls.cache = {}
        cls.tpch_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        cls.tpcds_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    mod.TestExecutor.setUpClass = classmethod(setUpClass)
    return True


def build(out: Path = CACHE) -> None:
    """Run the pinned test code once and keep what does not depend on sqlglot."""
    import unittest  # noqa: F401 - sqlglot's UNITTEST flag, as in the tests

    sys.path.insert(0, "/opt/sqlglot")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import duckdb  # noqa: F401
    import verify
    verify.cap_duckdb()  # as in the verifier children
    import tests.test_executor as te

    captured = []
    real_table = te.Table
    te.Table = lambda columns, rows: captured.append((columns, rows)) or real_table(columns=columns, rows=rows)
    try:
        te.TestExecutor.setUpClass()
    finally:
        te.Table = real_table
    cls = te.TestExecutor
    order = [(name, t, cols) for name, _, schema_attr, _, _ in DATASETS
             for t, cols in getattr(te, schema_attr).items()]
    assert len(order) == len(captured), (len(order), len(captured))
    rows = {}
    for (name, t, cols), (c2, r) in zip(order, captured):
        assert cols == c2, (name, t)
        rows[(name, t)] = r

    duck, seen = {}, {}
    for name, conn, sqls, with_opt in (("tpch", cls.tpch_conn, cls.tpch_sqls, True),
                                       ("tpcds", cls.tpcds_conn, cls.tpcds_sqls, False)):
        for _, sql, optimized in sqls:
            for src in ([sql, optimized] if with_opt else [sql]):
                try:
                    q = te.transpile(src, write="duckdb")[0]
                    a = conn.execute(q).fetchdf()
                    b = conn.execute(q).fetchdf()  # must be reproducible
                except Exception:  # noqa: BLE001 - not cached: runs (and fails) as usual
                    continue
                if not a.equals(b) or (q in seen and seen[q] != name):
                    duck.pop(q, None)
                    seen[q] = None
                    continue
                if q not in seen:
                    duck[q] = a
                    seen[q] = name
    cls.tearDownClass()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"rows": rows, "duck": duck}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"executor cache: {len(rows)} tables, {sum(len(r) for r in rows.values())} rows, "
          f"{len(duck)} DuckDB results, {out.stat().st_size >> 10} kB", file=sys.stderr)


if __name__ == "__main__":
    build()
