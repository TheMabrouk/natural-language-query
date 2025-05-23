import os
import re
import duckdb
import base64
import openai
import pandas as pd
from io import BytesIO
from django.conf import settings
from django.shortcuts import render
from .interface import SqlQueryGenerator, SqlChartGenerator
import matplotlib.pyplot as plt

# Load the Excel file once at startup
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "database.xlsx")
dfs = pd.read_excel(DATA_PATH, sheet_name=None)  # dict of sheets

client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=settings.GITHUB_TOKEN,
)
sql_generator = SqlQueryGenerator(client)
sql_generator.load_schema_from_dataframes(dfs)
chart_generator = SqlChartGenerator(client)

# Optional: Load into duckdb memory
con = duckdb.connect(database=":memory:")
for name, data in dfs.items():
    con.register(name, data)


def is_safe_sql(sql_query):
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for keyword in dangerous_keywords:
        if f" {keyword} " in f" {sql_query.upper()} ":
            return False
    return True


def preprocess_query(query: str) -> str:
    query = re.sub(
        r"(CURRENT_DATE\s*-\s*INTERVAL\s*'[\d\s\w]+')", r"CAST(\1 AS DATE)", query
    )
    query = re.sub(
        r"EXTRACT\((YEAR|MONTH|DAY|HOUR|MINUTE|SECOND) FROM ([a-zA-Z0-9_\.]+)\)",
        r"EXTRACT(\1 FROM CAST(\2 AS DATE))",
        query,
    )
    query = re.sub(
        r"([a-zA-Z0-9_\.]+)\s*(=|>|<|>=|<=|!=)\s*'(\d{4}-\d{2}-\d{2})'",
        r"CAST(\1 AS DATE) \2 DATE '\3'",
        query,
    )
    query = re.sub(
        r"(DATE_TRUNC|DATE_PART)\s*\(\s*'([^']+)'\s*,\s*([a-zA-Z0-9_\.]+)\s*\)",
        r"\1('\2', CAST(\3 AS DATE))",
        query,
    )
    query = re.sub(
        r"(CAST\([a-zA-Z0-9_\.]+\s*AS\s*BIGINT\)\s*-*\s*CAST\([a-zA-Z0-9_\.]+ AS BIGINT\))",
        lambda m: m.group(0).replace(
            "CAST", "CAST(CAST"
        ),  # Wrap timestamps in double CAST
        query,
    )
    query = re.sub(
        r"(JOIN\s+[a-zA-Z0-9_]+\s+ON\s+[a-zA-Z0-9_\.]+\s*=\s*)([a-zA-Z0-9_\.]+)(\s+(?:AND|OR|WHERE))",
        lambda m: (
            f"{m.group(1)}CAST({m.group(2)} AS DATE){m.group(3)}"
            if any(
                date_word in m.group(0).lower()
                for date_word in ["date", "time", "day", "month", "year"]
            )
            else m.group(0)
        ),
        query,
    )
    return query


def index(request):
    result = None
    query = ""
    chart_img = ""
    result_html = ""

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        if not question or len(question) < 5 or len(question) > 200:
            result_html = "<div class='error'>Please enter a valid question (5-200 characters).</div>"
            return render(request, "index.html", {"result": result_html})

        query = sql_generator.generate_sql(question, model="gpt-4o-mini")

        if not is_safe_sql(query):
            result_html = "<div class='error'>Unsafe SQL query detected. Please refine your question.</div>"
            return render(request, "index.html", {"result": result_html})

        try:
            query = preprocess_query(query)
            df = con.execute(query).df()
            if len(df) > 10:
                df_display = df.head(10)
                result_html = (
                    "<p><strong>Note:</strong> Display limited to 10 rows</p>"
                    + df_display.to_html(classes="table table-striped", index=False)
                )
            else:
                result_html = df.to_html(classes="table table-striped", index=False)

            sample = df.head(5).to_dict(orient="records")

            chart_code = chart_generator.generate_chart_code(sample, question)

            if chart_code.startswith("```python"):
                chart_code = chart_code[9:]
            elif chart_code.startswith("```"):
                chart_code = chart_code[3:]
            if chart_code.endswith("```"):
                chart_code = chart_code[:-3]

            if chart_code:

                import matplotlib

                matplotlib.use("Agg")

                namespace = {
                    "pd": pd,
                    "sns": __import__("seaborn"),
                    "plt": plt,
                    "sample_json": sample,
                }

                df_safe = df.copy()
                for col in df_safe.select_dtypes(include=["datetime"]):
                    df_safe[col] = df_safe[col].dt.strftime("%Y-%m-%d")

                chart_code = chart_code.replace(
                    "df = pd.DataFrame(sample_json)",
                    f"df = pd.DataFrame({df_safe.to_dict(orient='records')})",
                )

                chart_code = chart_code.replace("plt.show()", "")
                chart_code = chart_code + "\nplt.tight_layout()"

                exec(chart_code, namespace)

                buf = BytesIO()
                namespace["plt"].savefig(buf, format="png")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                chart_img = f"data:image/png;base64,{b64}"
                namespace["plt"].close()
            else:
                chart_img = ""

        except Exception as e:
            result_html = f"<div class='error'>Error: {e}</div>"

    return render(
        request,
        "index.html",
        {
            "query": query,
            "result": result_html,
            "chart_img": chart_img,
        },
    )
