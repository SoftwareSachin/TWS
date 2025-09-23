import { showSuccess } from "@/utils/toastUtils";
import Image from "next/image";
import copy from "../../../public/assets/chat/copy.svg";

interface APIClientSecretModalProps {
  secret: string;
  close: (val: string) => void;
}

const APIClientSecretModal = ({ secret, close }: APIClientSecretModalProps) => {
  const copySecret = () => {
    navigator.clipboard.writeText(secret);
    showSuccess("Secret copied to clipboard!");
  };
  return (
    <div className="space-y-6 pt-4">
      <div className="px-4">
        <div className="flex items-center justify-between bg-gray-100 border rounded-md px-3 py-2">
          <span className="text-sm break-all text-gray-800">{secret}</span>
          <Image
            src={copy}
            alt="copy"
            onClick={() => copySecret()}
            className="self-start cursor-pointer"
          />
        </div>

        <p className="text-red-500 text-sm mt-3 leading-relaxed">
          ⚠️ You will not be able to view this secret again. Please copy and
          store it securely.
        </p>
      </div>

      <div className="flex justify-end gap-3 px-4 py-3 border-t bg-gray-50 rounded-b-md">
        <button
          type="button"
          onClick={() => close("")}
          className="px-4 py-2 border rounded-md"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default APIClientSecretModal;
