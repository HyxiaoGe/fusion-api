from __future__ import annotations

import re
import sys
from pathlib import Path


TABLES = ("model_sources", "model_credentials")


def extract_copy_block(sql_text: str, table_name: str) -> str:
    pattern = re.compile(
        rf"^COPY public\.{table_name} \((.*?)\) FROM stdin;\n(.*?)\n\\\.$",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(sql_text)
    if not match:
        raise ValueError(f"Could not find COPY block for table: {table_name}")

    columns = match.group(1)
    body = match.group(2)
    return f"COPY public.{table_name} ({columns}) FROM stdin;\n{body}\n\\.\n"


def build_seed_sql(sql_text: str) -> str:
    blocks = [extract_copy_block(sql_text, table_name) for table_name in TABLES]

    return (
        "BEGIN;\n"
        "TRUNCATE TABLE public.model_credentials, public.model_sources RESTART IDENTITY CASCADE;\n\n"
        + "\n".join(blocks)
        + "\n"
        + "SELECT setval('public.model_sources_id_seq', COALESCE((SELECT MAX(id) FROM public.model_sources), 1), "
        + "(SELECT COUNT(*) > 0 FROM public.model_sources));\n"
        + "SELECT setval('public.model_credentials_id_seq', COALESCE((SELECT MAX(id) FROM public.model_credentials), 1), "
        + "(SELECT COUNT(*) > 0 FROM public.model_credentials));\n"
        + "COMMIT;\n"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python extract_model_seed.py <backup.sql> <output.sql>", file=sys.stderr)
        return 1

    backup_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not backup_path.exists():
        print(f"Backup file not found: {backup_path}", file=sys.stderr)
        return 1

    sql_text = backup_path.read_text(encoding="utf-8", errors="ignore")
    seed_sql = build_seed_sql(sql_text)
    output_path.write_text(seed_sql, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
