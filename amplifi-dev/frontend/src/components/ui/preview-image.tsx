import React, { useEffect, useState } from "react";
import { downloadApi } from "@/api/chatApp";
import Image from "next/image";
import Modal from "@/components/forms/modal";
import AddUserForm from "@/components/forms/addUserForm";

interface PreviewImageProps {
  file: any;
}

export const PreviewImage: React.FC<PreviewImageProps> = ({ file }) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const handlePreview = async () => {
    await downloadApi(file, setPreviewUrl);
  };
  const [expand, setIsExpanded] = useState(false);

  useEffect(() => {
    if (file?.file_name) {
      handlePreview();
    }
  }, [file]);
  return (
    <div>
      {previewUrl && (
        <>
          <div className="relative w-[35%] group">
            {/* Image */}
            <img
              src={previewUrl}
              alt={file.file_name}
              className="rounded-xl w-full h-[300px] transition duration-300 group-hover:blur-[2px]"
            />

            {/* Expand Icon */}
            <div
              title={"Expand Image"}
              onClick={() => setIsExpanded(true)}
              className="absolute right-2 top-2 rounded-full bg-[#9ba4f8] p-1 opacity-0 group-hover:opacity-100 transition duration-200"
            >
              <Image
                className="cursor-pointer"
                src="/assets/icons/expand-img.svg"
                alt="Expand"
                height={24}
                width={24}
              />
            </div>
          </div>
          <Modal
            isOpen={expand}
            onClose={() => {
              setIsExpanded(false);
            }}
            title={`${file.file_name}`}
            size={"!max-w-2xl"}
          >
            <img
              src={previewUrl}
              alt={file.file_name}
              className="rounded-xl w-full h-full"
            />
          </Modal>
        </>
      )}
    </div>
  );
};
