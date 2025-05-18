# utils.py
import re
from field_descriptions import field_descriptions
from sqlalchemy import create_engine, inspect
from langchain_community.utilities.sql_database import SQLDatabase
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
load_dotenv()
# ———————————————————————————
# 2) Variables de conexión
# ———————————————————————————
user      = os.getenv("HANA_USER")
pwd       = os.getenv("HANA_PWD")
host      = os.getenv("HANA_HOST")
port      = os.getenv("HANA_PORT")
schema    = os.getenv("HANA_SCHEMA")
view_name = os.getenv("HANA_VIEW")
user_enc = quote_plus(user)
pwd_enc = quote_plus(pwd)

uribd    = f"hana+hdbcli://{user_enc}:{pwd_enc}@{host}:{port}/?currentSchema={schema}"

engine = create_engine(uribd)
db_data = SQLDatabase.from_uri(uribd)
inspector = inspect(engine)

print("Connecting to:", uribd)

views = inspector.get_view_names(schema=schema)

def get_schema(_):
    cols = inspector.get_columns(view_name, schema=schema)
    lines = []
    for c in cols:
        name_columns = c["name"].upper()
        description_columns = field_descriptions.get(name_columns, "Descripción no disponible")
        lines.append(f"- {name_columns}: {description_columns}")
    return "\n".join(lines)


def run_query(query: str):
    query = re.sub(r'^```(?:sql)?\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'```$', '', query)
    query = re.sub(r'^"""{0,1}sql\s*', '', query, flags=re.IGNORECASE)
    query = re.sub(r'"""$', '', query)
    query = query.replace("`", '"')
    clean_sql = query.strip()
    return db_data.run(clean_sql)


def get_field_desc(_):
    lines = []
    for col, desc in field_descriptions.items():
        lines.append(f"- {col}: {desc}")
    return "\n".join(lines)