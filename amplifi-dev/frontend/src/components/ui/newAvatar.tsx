import { newAvatarProps } from "@/types/props/NewAvatarProps";
import clsx from "clsx";
import React from "react";

const colorMap = [
  { bg: "bg-emerald-100", text: "text-emerald-700" },
  { bg: "bg-indigo-100", text: "text-indigo-700" },
  { bg: "bg-purple-100", text: "text-purple-700" },
  { bg: "bg-pink-100", text: "text-pink-700" },
  { bg: "bg-yellow-100", text: "text-yellow-700" },
  { bg: "bg-blue-100", text: "text-blue-700" },
  { bg: "bg-red-100", text: "text-red-700" },
  { bg: "bg-green-100", text: "text-green-700" },
  { bg: "bg-teal-100", text: "text-teal-700" },
  { bg: "bg-cyan-100", text: "text-cyan-700" },
];

const NewAvatar: React.FC<
  newAvatarProps & React.HTMLAttributes<HTMLDivElement>
> = ({ title, className, ...rest }) => {
  const safeTitle = title || "AA";
  const initials = safeTitle
    .split(" ")
    .map((word) => word[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  const colorIndex = safeTitle.length % colorMap.length;
  const { bg, text } = colorMap[colorIndex];

  return (
    <div
      {...rest}
      className={clsx(
        `w-10 h-10 rounded-full outline outline-1 outline-offset-[-0.5px] outline-white flex justify-center items-center`,
        className,
        bg,
      )}
    >
      <span className={clsx("text-xs font-medium", text)}>{initials}</span>
    </div>
  );
};

export default NewAvatar;
