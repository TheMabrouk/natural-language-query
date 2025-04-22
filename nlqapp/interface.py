import os
import re
import json
import openai
import pandas as pd
from typing import Dict, List, Optional


class SqlQueryGenerator:
    def __init__(self, client):
        """
        Initialize the SQL query generator with OpenAI API key.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        self.client = client

        # Store schema information
        self.schema_info = {}

    def load_schema_from_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Load schema information from a dictionary of dataframes.

        Args:
            dataframes: Dictionary with sheet names as keys and dataframes as values
        """
        self.schema_info = {}

        for sheet_name, df in dataframes.items():
            columns = df.columns.tolist()
            sample_data = df.head(5).to_dict(orient="records")

            self.schema_info[sheet_name] = {
                "columns": columns,
                "sample_data": sample_data,
            }

    def get_schema_description(self) -> str:
        """Generate a text description of the database schema with clear column definitions."""
        description = "DATABASE SCHEMA:\n\n"

        for table_name, table_info in self.schema_info.items():
            description += f"Table: {table_name}\n"
            description += "Columns:\n"
            for col in table_info["columns"]:
                description += f"  - {col}\n"
            description += "Sample data row:\n"
            if table_info["sample_data"]:
                description += f"  {json.dumps(table_info['sample_data'][0])}\n"
            description += "\n"

        return description

    def generate_sql(self, query: str, model: str = "gpt-4.1") -> str:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language query
            model: OpenAI model to use

        Returns:
            SQL query string
        """
        schema_description = self.get_schema_description()

        system_prompt = f"""
        You are an expert SQL developer. Your task is to convert natural language queries into SQL queries for DuckDB.

        Use the following database schema EXACTLY as defined - do not invent column names:

        {schema_description}

        Important guidelines:
        1. Generate only SQL code, no explanations.
        2. Use proper SQL syntax for queries.
        3. VERIFY COLUMN NAMES: Double-check that you're using the exact column names from the schema.
        4. Make reasonable assumptions about relationships between tables when not explicitly stated.
        5. Handle table joins appropriately based on the schema.
        6. For fact tables, assume they can be joined to dimension tables using the corresponding key fields.
        7. Return only the SQL query without additional comments or markdown formatting.
        8. If you are uncertain about a column name, refer back to the schema definition.
        9. IMPORTANT: For date columns, use CAST(column AS DATE) before applying date functions.
        10. When using date functions like YEAR(), always ensure the column is properly cast as a DATE type.
        11. For window functions with OVER(): DO NOT use GROUP BY unless you're applying an aggregate function to the column first.
        12. For cumulative calculations, prefer: SUM(value) OVER (ORDER BY date_column) without GROUP BY.
        """

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this request to SQL: {query}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        sql_query = response.choices[0].message.content.strip()

        # Remove markdown code block formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()

        return sql_query

    def validate_query(self, sql_query):
        """Check if the SQL query uses valid column names from the schema, including alias handling."""
        # Build a mapping of lowercase column names to fully qualified names
        valid_columns = {}
        for table, info in self.schema_info.items():
            for col in info["columns"]:
                valid_columns[col.lower()] = f"{table}.{col}"

        # Step 1: Extract alias mappings
        alias_pattern = re.compile(
            r"\b(from|join)\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)",
            re.IGNORECASE,
        )
        aliases = {}
        for match in alias_pattern.findall(sql_query):
            _, table, alias = match
            aliases[alias.lower()] = table.lower()

        # Step 2: Find all table.column references
        query_lower = sql_query.lower()
        possible_columns = re.findall(
            r"([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", query_lower
        )

        issues = []
        for alias, col in possible_columns:
            if alias in aliases:
                actual_table = aliases[alias]
                table_info = self.schema_info.get(actual_table)
                if table_info:
                    valid_cols = [c.lower() for c in table_info["columns"]]
                    if col not in valid_cols:
                        suggestions = self.find_similar_columns(col, actual_table)
                        issues.append(
                            f"Column '{col}' not found in table '{actual_table}' (alias '{alias}'). "
                            f"Did you mean: {suggestions}?"
                        )
                else:
                    issues.append(
                        f"Table alias '{alias}' refers to unknown table '{actual_table}'."
                    )
            else:
                issues.append(f"Table alias '{alias}' not found in query context.")

        return issues

    def find_similar_columns(self, col_name, table_name):
        """Find similar column names in the given table."""
        if table_name not in self.schema_info:
            return "table not found"

        cols = [c.lower() for c in self.schema_info[table_name]["columns"]]
        return ", ".join(difflib.get_close_matches(col_name, cols, n=2, cutoff=0.6))


def load_excel_sheets_to_dataframes(excel_file_path):
    """Load all sheets from Excel file into a dictionary of dataframes."""
    excel_file = pd.ExcelFile(excel_file_path)
    dataframes = {}

    for sheet_name in excel_file.sheet_names:
        dataframes[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

    return dataframes


class SqlChartGenerator:
    def __init__(self, client):
        self.client = client

    def generate_chart_code(
        self, sample_records: list, query: str, model: str = "gpt-4.1"
    ) -> str:
        """
        Given a tiny sample of the query results as a list of dicts,
        ask the LLM to produce complete seaborn/matplotlib code
        that, when run, will plot that sample appropriately.
        """
        sample_json = json.dumps(sample_records, indent=2)
        prompt = f"""
        You are a Python data visualization expert. Here is a small JSON sample of my query result:

        {sample_json}

        The natural language question is: "{query}"

        Write a **complete** Python snippet that:
        1. ONLY imports from these allowed libraries: pandas as pd, matplotlib.pyplot as plt, and seaborn as sns.
        2. Loads the data directly using: df = pd.DataFrame(sample_json)
        3. Creates a visualization using ONLY matplotlib or seaborn standard plotting functions.
        4. Uses only core visualization functions like plt.figure(), plt.plot(), plt.bar(), sns.lineplot(), sns.barplot(), etc.
        5. Adds appropriate title, labels, and styling using plt.title(), plt.xlabel(), plt.ylabel().
        6. Includes plt.tight_layout() at the end.
        7. DO NOT include plt.show() or any file operations.
        8. DO NOT use any external libraries beyond pandas, matplotlib, and seaborn.
        9. DO NOT use any system, file, network, or advanced imports (like os, sys, requests, subprocess).
        10. DO NOT include any explanatory comments—just runnable code.

        If no sensible visualization is possible with this data, return an empty string.
        """
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Python data‑viz expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        code = resp.choices[0].message.content.strip()
        # save code to tmp file for checking
        with open("tmp_code.py", "w") as f:
            f.write(code)
        return code
