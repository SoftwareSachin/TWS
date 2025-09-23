import React from "react";
import Image from "next/image";

export const Icon = ({ iconName, size = 20 }) => {
  return (
    <Image
      width={size}
      height={size}
      alt={iconName}
      src={`/assets/chat/${iconName}.svg`}
    ></Image>
  );
};
