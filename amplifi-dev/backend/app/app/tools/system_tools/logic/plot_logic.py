import ast
import json

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from pydantic_ai.messages import BinaryContent, ToolReturn

from app.api.deps import get_async_gpt4o_client
from app.be_core.config import settings
from app.be_core.logger import logger
from app.tools.system_tools.schemas.plot_schema import (
    PlotMockInput,
    PlotMockOutput,
    PlotOutput,
)

client = get_async_gpt4o_client()


PLOT_PROMPT_TEMPLATE = """
You are a helpful assistant that generates JavaScript-compatible Plotly charts.

Given the user query: "{query}", generate the appropriate chart type using JavaScript-style Plotly code.

{dataframe_context}

IMPORTANT: Follow the user's query EXACTLY as stated. Do not make assumptions about data aggregation unless the user explicitly requests it

Respond ONLY with a JSON object like this:

CASE 1: If sample data is passed above (dataframe_context is not empty):
{{
  "plot_code": "fig = go.Figure(...)\\nfig.update_layout(...)",
  "plot_html": "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Chart</title><script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script></head><body><div id=\"plot\" style=\"height: 450px; width: 100%;\"></div><script>var data = PLACEHOLDER_DATA;\\nvar layout = PLACEHOLDER_LAYOUT; Plotly.newPlot('plot', data, layout);</script></body></html>"
}}

CASE 2: If no sample data is passed above, or data is available in user query (dataframe_context is empty):
{{
  "plot_code": "",
  "plot_html": "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\"><title>Participant Performance Data</title><script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script></head><body><div id=\"top-10-highest-lowest-priced-films\" style=\"height: 450px; width: 100%;\"></div><script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script><script>var data = [...];\\nvar layout = {{...}}; Plotly.newPlot('top-10-highest-lowest-priced-films', data, layout);</script></body></html>"
}}

CRITICAL RULES:

FOR CASE 1 (when dataframe_context is provided):
- `plot_code` MUST contain executable Python Plotly code
- `plot_code` MUST NOT CONTAIN ANY IMPORT STATEMENTS
- `plot_code` MUST create a go.Figure in variable 'fig' USING ONLY THE EXISTING PANDAS DATAFRAME NAMED df
- `plot_code` should NOT create, generate, or define any new data - ONLY use the columns and data from the existing df
- `plot_code` should NOT contain pd.DataFrame(), pd.read_csv(), or any data creation code
- `plot_code` should ONLY reference existing columns from df (e.g., df['column_name'])
- `plot_html` MUST contain PLACEHOLDER_DATA and PLACEHOLDER_LAYOUT exactly as written - DO NOT replace them with actual data
- `plot_html` should contain proper axis labels and titles

FOR CASE 2 (when dataframe_context is empty):
- `plot_code` MUST be empty string ""
- `plot_html` MUST contain actual JavaScript data arrays and layout objects
- `plot_html` should NOT contain PLACEHOLDER_DATA or PLACEHOLDER_LAYOUT
- `plot_html` should contain realistic sample data that matches the query

GENERAL RULES:
- Do NOT include markdown formatting (no ```json)
- Do NOT include explanations or comments — only valid JSON output
- The dataframe_context shows only 3 sample rows to help you understand the data structure (column names, data types)
- Always include proper axis labels and chart titles
- Make sure the chart type matches what the user requested

EXAMPLES:

For CASE 1 (with dataframe_context):
"var data = PLACEHOLDER_DATA;\\nvar layout = PLACEHOLDER_LAYOUT;"

For CASE 2 (without dataframe_context):
"var data = [{{\\"x\\": [\\"A\\", \\"B\\"], \\"y\\": [1, 2], \\"type\\": \\"bar\\"}}];\\nvar layout = {{\\"title\\": \\"Example Chart\\"}};"

Make sure:
- `plot_code` is generated ONLY if you are given the data structure of the df
- The dataframe_context shows only 3 sample rows to help you understand the data structure (column names, data types)
- The figure is saved as a `go.Figure` if you're referencing Python code.
- If you are generating `plot_code` you have access to a dataframe called df
- If you are generating `plot_code` there must be no import statements generated
"""


async def generate_plotly_from_llm(plot_input: PlotMockInput) -> PlotOutput:
    try:
        logger.info(f"[PLOT-GENERATE] Input Query: {plot_input.query}")
        logger.info(f"[PLOT-GENERATE] CSV File Name: {plot_input.csv_file_name}")

        # If we have a CSV file name, read the full dataset from the CSV export
        dataframe_data = None
        if plot_input.csv_file_name:
            try:
                csv_file_path = (
                    f"{settings.CSV_EXPORT_FOLDER}/{plot_input.csv_file_name}"
                )
                logger.info(
                    f"[PLOT-GENERATE] Reading full dataset from CSV: {csv_file_path}"
                )

                full_dataframe = pd.read_csv(csv_file_path).head(settings.MAX_PLOT_DATA)
                logger.info(
                    f"[PLOT-GENERATE] Full dataset loaded: {full_dataframe.shape[0]} rows, {full_dataframe.shape[1]} columns"
                )

                dataframe_data = full_dataframe.to_dict(orient="records")

            except Exception as csv_error:
                logger.warning(
                    f"[PLOT-GENERATE] Failed to read CSV file {plot_input.csv_file_name}: {csv_error}"
                )

        # Prepare dataframe context if available
        dataframe_context = ""
        if dataframe_data:
            sample_df = pd.DataFrame(dataframe_data).head(3)
            dataframe_context = f"\n\nDataframe context (showing 3 sample rows from full dataset):\n{sample_df.to_string()}"
            logger.info(
                f"[PLOT-GENERATE] Added dataframe context with {len(dataframe_data)} total rows"
            )
        else:
            logger.info(
                "[PLOT-GENERATE] No dataframe data available, proceeding with base prompt only"
            )

        # Format the prompt with both query and dataframe_context
        prompt = PLOT_PROMPT_TEMPLATE.format(
            query=plot_input.query, dataframe_context=dataframe_context
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You return JSON with Plotly code and its HTML output. You return ONLY valid JSON — do NOT wrap your response in markdown code blocks like ```json.",
                },
                {"role": "user", "content": prompt},
            ],  # using OpenAI's JSON mode
        )

        content = response.choices[0].message.content
        logger.info(f"[PLOT-GENERATE] LLM JSON Response: {content}")

        try:
            parsed = json.loads(content)
            plot_code = parsed.get("plot_code", "")
            plot_html = parsed.get("plot_html", "")

            if not plot_code and not plot_html:
                logger.warning(
                    "[PLOT-GENERATE] Missing plot_code and plot_html in LLM response"
                )
                return PlotMockOutput(
                    plot_code="",
                    plot_html="",
                    error="Missing plot_code and plot_html in LLM response",
                )

        except json.JSONDecodeError as e:
            logger.error(
                f"[PLOT-GENERATE] Failed to parse LLM response as JSON: {str(e)}"
            )
            return PlotMockOutput(
                plot_code="",
                plot_html="",
                error=f"Invalid JSON response from LLM: {str(e)}",
            )

        # Execute the plot code if we have dataframe data
        if plot_code and dataframe_data:
            try:
                logger.info(
                    "[PLOT-GENERATE] Executing Python plot code with dataframe data"
                )

                # Convert input to pandas DataFrame
                dataframe = pd.DataFrame(dataframe_data)

                # Execute the plot code with df/go/pd available
                local_vars = {"df": dataframe, "go": go, "pd": pd, "px": px}

                local_vars = safe_execute_plot_code(plot_code, local_vars)

                if "fig" in local_vars:
                    fig = local_vars["fig"]
                    logger.info("[PLOT-GENERATE] Successfully created go.Figure object")

                    # Replace placeholders in plot_html with actual data from fig
                    final_plot_html = plot_html
                    if (
                        plot_html
                        and "PLACEHOLDER_DATA" in plot_html
                        and "PLACEHOLDER_LAYOUT" in plot_html
                    ):
                        try:
                            fig.layout.template = None
                            json_data = json.dumps(
                                {"data": fig.data, "layout": fig.layout},
                                cls=plotly.utils.PlotlyJSONEncoder,
                            )
                            data_dict = json.loads(json_data)

                            final_plot_html = plot_html.replace(
                                "PLACEHOLDER_DATA", str(data_dict["data"])
                            )
                            final_plot_html = final_plot_html.replace(
                                "PLACEHOLDER_LAYOUT", str(data_dict["layout"])
                            )
                            logger.info(
                                "[PLOT-GENERATE] Replaced placeholders with actual fig data"
                            )
                        except Exception as placeholder_err:
                            logger.warning(
                                f"[PLOT-GENERATE] Failed to replace placeholders: {placeholder_err}"
                            )
                    else:
                        logger.info(
                            "[PLOT-GENERATE] Could not find PLACEHOLDER_DATA and PLACEHOLDER_LAYOUT in plot_html"
                        )

                    # Try to export an in-memory PNG
                    image_bytes: bytes | None = None
                    bin_content = None
                    try:
                        image_bytes = fig.to_image(
                            format="png", width=1000, height=800, scale=3
                        )
                        logger.info(
                            f"[PLOT-GENERATE] Generated plot image in memory, size: {len(image_bytes)} bytes"
                        )

                        bin_content = BinaryContent(
                            data=image_bytes, media_type="image/png"
                        )
                    except Exception as img_err:
                        logger.warning(
                            f"[PLOT-GENERATE] Failed to render PNG in-memory: {img_err}"
                        )

                    return ToolReturn(
                        return_value={
                            "plot_code": plot_code,
                            "plot_html": final_plot_html,
                        },
                        content=[
                            "Here is the generated plot:",
                            bin_content,
                            "Please use this image to answer any follow-up questions if the user asks about the plot",
                        ],
                    )

                else:
                    logger.warning(
                        "[PLOT-GENERATE] No 'fig' variable found in executed code"
                    )

            except Exception as e:
                logger.error(f"[PLOT-GENERATE] Error executing plot code: {str(e)}")

        # Return the original response if no execution or execution failed
        return PlotMockOutput(plot_code=plot_code, plot_html=plot_html)

    except Exception as e:
        logger.error(f"[PLOT-GENERATE] Error: {str(e)}")
        return PlotMockOutput(plot_code="", plot_html="", error=str(e))


def safe_execute_plot_code(plot_code: str, local_vars: dict) -> dict:
    """
    Safely execute plot code using AST parsing and compile/eval instead of exec.
    This approach is safer as it validates the syntax before execution.
    """
    try:
        # Parse the code into an AST to validate syntax
        tree = ast.parse(plot_code)

        # Check for potentially dangerous operations
        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")

            # Block file operations
            if isinstance(node, ast.Call):
                if hasattr(node.func, "id"):
                    if node.func.id in [
                        "open",
                        "file",
                        "input",
                        "raw_input",
                        "eval",
                        "exec",
                    ]:
                        raise ValueError(
                            f"Dangerous function '{node.func.id}' is not allowed"
                        )
                elif hasattr(node.func, "attr"):
                    if node.func.attr in ["open", "read", "write", "remove", "delete"]:
                        raise ValueError(
                            f"Dangerous method '{node.func.attr}' is not allowed"
                        )

        # Compile the code
        compiled_code = compile(tree, "<plot_code>", "exec")

        exec(compiled_code, {}, local_vars)  # nosec

        return local_vars

    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in plot code: {e}")
    except Exception as e:
        raise ValueError(f"Error executing plot code: {e}")
