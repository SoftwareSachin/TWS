"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import searchIcon from "@/assets/icons/search-icon.svg";

interface SearchBoxProps {
  placeholder?: string;
  value: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDebouncedChange?: (value: string) => void;
  debounceDelay?: number;
  className?: string;
  inputClassName?: string;
  autoFocus?: boolean;
  name?: string;
}

const SearchBox: React.FC<SearchBoxProps> = ({
  placeholder = "Search here",
  value,
  onChange,
  onDebouncedChange,
  debounceDelay = 300,
  className = "",
  inputClassName = "",
  autoFocus = false,
  name = "search",
}) => {
  const [internalValue, setInternalValue] = useState(value);

  // Update internal value when external value changes
  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  // Debounce effect
  useEffect(() => {
    const handler = setTimeout(() => {
      if (onDebouncedChange) {
        onDebouncedChange(internalValue);
      }
    }, debounceDelay);

    return () => {
      clearTimeout(handler);
    };
  }, [internalValue, debounceDelay, onDebouncedChange]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInternalValue(e.target.value);
    onChange?.(e); // Trigger immediate onChange if needed
  };

  return (
    <label
      htmlFor={name}
      className={`bg-white px-2 py-1 flex items-center gap-2 rounded-lg border border-gray-300 shadow-sm focus-within:ring-2 focus-within:ring-blue-500 ${className}`}
    >
      <Image src={searchIcon} alt="Search" width={20} height={20} />
      <input
        id={name}
        name={name}
        type="text"
        placeholder={placeholder}
        value={internalValue}
        onChange={handleChange}
        autoFocus={autoFocus}
        aria-label={placeholder}
        className={`bg-transparent outline-none w-full text-sm ${inputClassName}`}
      />
    </label>
  );
};

export default SearchBox;
