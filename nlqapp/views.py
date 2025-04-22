import os
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


def index(request):
    result = None
    query = ""
    chart_img = ""
    result_html = ""
    if request.method == "POST":
        question = request.POST.get("question")
        query = sql_generator.generate_sql(question, model="gpt-4o-mini")

        try:
            df = con.execute(query).df()
            result_html = df.to_html(classes="table table-striped", index=False)

            sample = df.head(5).to_dict(orient="records")
            chart_code = chart_generator.generate_chart_code(sample, question)

            if chart_code:
                # Set non-interactive matplotlib backend to prevent window opening
                import matplotlib

                matplotlib.use("Agg")

                namespace = {
                    "pd": pd,
                    "sns": __import__("seaborn"),
                    "plt": plt,
                    "json": __import__("json"),
                }
                namespace["sample_json"] = sample
                chart_code = chart_code.replace(
                    "df = pd.DataFrame(...)", "df = pd.DataFrame(sample_json)"
                )

                chart_code = chart_code.replace("plt.show()", "")
                # add tight_layout
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
