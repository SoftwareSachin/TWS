import { Button } from "../ui/button";

interface ConfirmModalProps {
  message: string;
  isLoading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmModal = ({
  message,
  isLoading,
  onConfirm,
  onCancel,
}: ConfirmModalProps) => {
  return (
    <div className="space-y-6 pt-4">
      <div className="px-4">
        <p className="text-gray-700">{message}</p>
      </div>

      <div className="flex justify-end space-x-2 p-4 border-t">
        <button
          type="button"
          onClick={() => onCancel()}
          className="px-4 py-2 border rounded-md"
        >
          Cancel
        </button>
        <Button
          isLoading={isLoading}
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-md"
          onClick={() => onConfirm()}
        >
          Confirm
        </Button>
      </div>
    </div>
  );
};

export default ConfirmModal;
