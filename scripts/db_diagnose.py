from __future__ import annotations

from sqlalchemy import inspect, text

from db import (
    database_url_warnings,
    engine,
    redacted_database_url,
    schema_compatibility_issues,
)


def main() -> int:
    print(f"database_url={redacted_database_url()}")
    warnings = database_url_warnings()
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("warnings: none")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("connection: ok")
    except Exception as error:
        print(f"connection: error ({type(error).__name__}) {error}")
        return 1

    try:
        inspector = inspect(engine)
        tables = sorted(inspector.get_table_names())
        print(f"tables: {', '.join(tables) if tables else '(none)'}")
        required = {
            "teachers",
            "classrooms",
            "students",
            "rounds",
            "attendance",
            "matches",
            "homework_entries",
        }
        missing = sorted(required - set(tables))
        if missing:
            print(f"missing_tables: {', '.join(missing)}")
            return 2
        print("missing_tables: none")
    except Exception as error:
        print(f"schema_check: error ({type(error).__name__}) {error}")
        return 3

    try:
        issues = schema_compatibility_issues()
        if issues:
            print("schema_compatibility: issues")
            for issue in issues:
                print(f"- {issue}")
            return 4
        print("schema_compatibility: ok")
    except Exception as error:
        print(f"schema_compatibility: error ({type(error).__name__}) {error}")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
