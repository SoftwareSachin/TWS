export const extractTableData = (html: string): Array<Array<string>> | null => {
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    const table = doc.querySelector("table");

    if (!table) return null;

    const rows: Array<Array<string>> = [];
    const tableRows = table.querySelectorAll("tr");

    tableRows.forEach((row) => {
      const cells: string[] = [];
      const tableCells = row.querySelectorAll("td, th");
      tableCells.forEach((cell) => {
        cells.push(cell.textContent?.trim() || "");
      });
      if (cells.length > 0) {
        rows.push(cells);
      }
    });

    return rows.length > 0 ? rows : null;
  } catch (error) {
    console.error("Error parsing table HTML:", error);
    return null;
  }
};

export const convertTableToText = (html: string): string => {
  const tableData = extractTableData(html);
  if (!tableData) {
    return html.replace(/<[^>]*>/g, "").trim();
  }

  let textOutput = "";
  tableData.forEach((row, index) => {
    if (index === 0) {
      textOutput += row.join(" | ") + "\n";
      textOutput += row.map(() => "---").join(" | ") + "\n";
    } else {
      textOutput += row.join(" | ") + "\n";
    }
  });

  return textOutput.trim();
};
