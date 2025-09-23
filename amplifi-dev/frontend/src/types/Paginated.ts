export interface PaginatedResponse<T> extends ListResults<T> {
  total: number;
  page: number;
  size: number;
  pages: number;
  previous_page: number;
  next_page: number;
}
export interface ListResults<T> {
  items: T[];
}

export interface Page {
  page: number;
  size: number;
}
