export interface ApiResponse<T> {
  message: string;
  meta: Record<string, any>;
  data: T;
}
