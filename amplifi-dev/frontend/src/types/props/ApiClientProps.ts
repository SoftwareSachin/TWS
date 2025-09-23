import { ApiClientItem } from "../ApiClient";

export interface AddApiClientFormProps {
  onSave: (clientSecret?: string) => void;
  setIsOpen: (val: boolean) => void;
  apiClientData?: ApiClientItem;
}
