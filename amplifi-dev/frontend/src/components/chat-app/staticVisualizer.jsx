import React, { useEffect, useRef } from "react";

const StaticVisualizer = () => {
  const visualContainerRef = useRef(null);
  const visualValueCount = 30;
  const mode = "speaker";
  // Generate the array for count 30

  useEffect(() => {
    if (!visualContainerRef.current) return;
    const visualContainer = visualContainerRef.current;
    let elements = [];
    if (!document.getElementsByClassName(mode).length) {
      for (let i = 0; i < visualValueCount; i++) {
        const div = document.createElement("div");
        div.className = "chat-visualizer-line " + mode;
        visualContainer.appendChild(div);
        elements.push(div);
      }
    } else {
      elements = Array.from(document.getElementsByClassName(mode)).filter(
        (el) => el instanceof HTMLDivElement,
      );
    }
    const processFrame = () => {
      elements.forEach((elm) => {
        const value = Math.random() * 0.5;
        const elmStyles = elm.style;
        elmStyles.transform = `scaleY(${value})`;
        elmStyles.opacity = `${Math.max(0.25, value)}`;
        elmStyles.background = `red`;
      });
      setTimeout(() => processFrame(), 200);
    };
    processFrame();
  }, []); // Only run this effect when isMicActive changes

  return (
    <div
      ref={visualContainerRef}
      className={"flex justify-between flex-1 items-center"}
    >
      {/* Visual elements will be dynamically created here */}
    </div>
  );
};

export default StaticVisualizer;
