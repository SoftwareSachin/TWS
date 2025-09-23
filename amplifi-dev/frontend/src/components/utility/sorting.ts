import { constants, SortDirection } from "@/lib/constants";

/**
 * Handles sort state logic
 * @param {string} field - field clicked
 * @param {string|null} sortField - current sort field
 * @param {string|null} sortDirection - current sort direction
 * @returns {{ nextField: string|null, nextDirection: string|null }}
 */

export const getNextSortState = (
  field: string,
  sortField: string | null,
  sortDirection: string | null,
): { nextField: string | null; nextDirection: string | null } => {
  if (sortField === field) {
    if (sortDirection === SortDirection.ASCENDING) {
      return { nextField: field, nextDirection: SortDirection.DESCENDING };
    } else if (sortDirection === SortDirection.DESCENDING) {
      return { nextField: null, nextDirection: null };
    }
  }
  return { nextField: field, nextDirection: SortDirection.ASCENDING };
};

/**
 * Generic sorting function
 * @param {Array} data - array to sort
 * @param {string} sortField - field to sort on
 * @param {string} sortDirection - asc/desc
 * @param {string[]} fields - special field order (like ["name", "count", "date"])
 */
export const getSortedData = (
  data: Array<any>,
  sortField: string,
  sortDirection: string,
) => {
  if (!sortField || !sortDirection) return data;
  const normalize = (v: any) => (v ?? "").toString().toLowerCase();

  const allSame = data.every(
    (item) => normalize(item[sortField]) === normalize(data[0][sortField]),
  );
  if (allSame) return data;

  return [...data].sort((a, b) => {
    let aValue = a[sortField];
    let bValue = b[sortField];

    // 🔹 normalize based on type
    if (!isNaN(Number(aValue)) && !isNaN(Number(bValue))) {
      // numeric
      aValue = Number(aValue);
      bValue = Number(bValue);
    } else if (Date.parse(aValue) && Date.parse(bValue)) {
      // date
      aValue = new Date(aValue).getTime();
      bValue = new Date(bValue).getTime();
    } else {
      // fallback → string
      aValue = (aValue ?? "").toString().toLowerCase();
      bValue = (bValue ?? "").toString().toLowerCase();
    }

    if (sortDirection === SortDirection.ASCENDING) {
      return aValue > bValue ? 1 : -1;
    } else {
      return aValue < bValue ? 1 : -1;
    }
  });
};
