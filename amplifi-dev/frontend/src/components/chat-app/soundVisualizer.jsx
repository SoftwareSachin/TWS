import React, { useEffect, useRef, useState } from "react";

const SoundVisualizer = () => {
  const visualContainerRef = useRef(null);
  const [audioContext, setAudioContext] = useState(null);
  const visualValueCount = 30;
  const mode = "mic";
  const generateDataMap = (count) => {
    const mid = Math.floor(count / 2);
    const firstHalf = Array.from({ length: mid }, (_, i) => mid - 1 - i); // Descending order
    const secondHalf = Array.from({ length: mid }, (_, i) => i); // Ascending order
    return [...firstHalf, ...secondHalf];
  };

  // Generate the array for count 30
  const dataMap = generateDataMap(visualValueCount);

  useEffect(() => {
    if (!visualContainerRef.current) return;
    if (audioContext?.state === "closed") {
      audioContext.resume();
    }
    // Initialize visual elements when mic becomes active
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

    // Audio visualization function
    const processFrame = (data) => {
      const values = Array.from(data);
      elements.forEach((elm, i) => {
        const value = values[dataMap[i]] / 255;
        const elmStyles = elm.style;
        elmStyles.transform = `scaleY(${value})`;
        elmStyles.opacity = `${Math.max(0.25, value)}`;
        elmStyles.background = `red`;
      });
    };
    const newAudioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    setAudioContext(newAudioContext);
    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        const analyser = newAudioContext.createAnalyser();
        analyser.smoothingTimeConstant = 0.5;
        analyser.fftSize = 32;

        const source = newAudioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        const frequencyData = new Uint8Array(analyser.frequencyBinCount);

        const renderFrame = () => {
          analyser.getByteFrequencyData(frequencyData);
          processFrame(frequencyData);
          requestAnimationFrame(renderFrame);
        };

        renderFrame();
      } catch (error) {
        console.error("Microphone access failed:", error);
      }
    };

    initAudio();

    // Cleanup visualizer and audio context when mic is inactive
    return () => {
      // Stop the visualizer and cleanup
      if (audioContext && audioContext.state !== "closed") {
        audioContext.close();
        Array.from(
          document.getElementsByClassName("chat-visualizer-line"),
        ).forEach((el) => {
          el.removeAttribute("style");
        }); // Close the audio context when inactive
      }
    };
  }, [audioContext]); // Only run this effect when isMicActive changes

  return (
    <div
      ref={visualContainerRef}
      className={"flex justify-between flex-1 items-center"}
    >
      {/* Visual elements will be dynamically created here */}
    </div>
  );
};

export default SoundVisualizer;
