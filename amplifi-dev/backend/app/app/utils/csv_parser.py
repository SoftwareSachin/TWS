from typing import List, Set


## check if a str from a set of acceptable strings is a column name
def find_csv_column(df_columns: List[str], allowed_strs: Set[str]) -> bool:

    for column in df_columns:
        for alias in allowed_strs:
            if column.strip().lower() == alias.lower():
                return column

    return False
