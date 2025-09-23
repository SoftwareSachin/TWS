/**
 * Filters a list of objects based on a given key and search value.
 *
 * @param list - The array of objects to filter
 * @param key - The key to search on (string only)
 * @param searchValue - The string to match (case-insensitive)
 * @returns A filtered array where the key's value includes the search value
 */
export function filterByKey<T extends Record<string, any>>(
  list: T[],
  key: keyof T,
  searchValue: string,
): T[] {
  const searchLower = searchValue.toLowerCase();

  return list.filter((item) => {
    const value: any = item[key];

    if (typeof value === "string") {
      return value.toLowerCase().includes(searchLower);
    }

    if (typeof value === "number" || typeof value === "boolean") {
      return String(value).toLowerCase().includes(searchLower);
    }

    return false; // skip if value is object/array/etc.
  });
}
