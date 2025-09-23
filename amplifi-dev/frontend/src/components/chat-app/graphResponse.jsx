"use client";

import React, { useEffect, useRef } from "react";
import * as Plotly from "plotly.js-dist";

const GraphResponseComponent = ({ responseText }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!responseText || !chartRef.current) return;

    const extractAndExecuteGraph = (text) => {
      try {
        const scriptRegex = /<script[^>]*>([\s\S]*?)<\/script>/gi;
        const scripts = [];
        let match;

        while ((match = scriptRegex.exec(text)) !== null) {
          scripts.push(match[1]);
        }

        // Find the script that contains Plotly code
        const plotlyScript = scripts.find(
          (script) =>
            script.includes("Plotly.newPlot") || script.includes("plotly"),
        );

        if (plotlyScript) {
          const dataMatch = plotlyScript.match(
            /var\s+data\s*=\s*(\[[\s\S]*?\]);/,
          );
          const layoutMatch = plotlyScript.match(
            /var\s+layout\s*=\s*(\{[\s\S]*?\});/,
          );

          if (dataMatch && layoutMatch) {
            const data = eval("(" + dataMatch[1] + ")");
            let layout;
            try {
              layout = eval("(" + layoutMatch[1] + ")");

              if (layout.title) {
                layout.title = {
                  text: layout.title?.text || layout.title,
                  font: { size: 16 },
                };
              }

              layout.xaxis = {
                ...layout.xaxis,
                title: {
                  text:
                    layout.xaxis?.title?.text ||
                    layout.xaxis?.title ||
                    "X Axis",
                  font: { size: 14 },
                },
                automargin: true,
              };

              layout.yaxis = {
                ...layout.yaxis,
                title: {
                  text:
                    layout.yaxis?.title?.text ||
                    layout.yaxis?.title ||
                    "Y Axis",
                  font: { size: 14 },
                },
                automargin: true,
              };

              layout.yaxis2 = {
                ...layout.yaxis2,
                title: {
                  text:
                    layout.yaxis2?.title?.text ||
                    layout.yaxis2?.title ||
                    "Y2 Axis",
                  font: { size: 14 },
                },
                automargin: true,
              };
            } catch (e) {
              console.error("Error parsing layout:", e);
            }

            // Create the plot
            Plotly.newPlot(chartRef.current, data, layout, {
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
              modeBarButtonsToAdd: [
                "drawline",
                "drawcircle",
                "drawrect",
                "drawopenpath",
                "eraseshape",
              ],
            });

            return true;
          }
        }
        const dataPattern = /data\s*=\s*\[([\s\S]*?)\]/;
        const layoutPattern = /layout\s*=\s*\{([\s\S]*?)\}/;

        const dataPatternMatch = text.match(dataPattern);
        const layoutPatternMatch = text.match(layoutPattern);

        if (dataPatternMatch && layoutPatternMatch) {
          const data = eval("([" + dataPatternMatch[1] + "])");
          const layout = eval("({" + layoutPatternMatch[1] + "})");

          Plotly.newPlot(chartRef.current, data, layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
            modeBarButtonsToAdd: [
              "drawline",
              "drawcircle",
              "drawrect",
              "drawopenpath",
              "eraseshape",
            ],
          });

          return true;
        }

        return false;
      } catch (error) {
        console.error("Error parsing graph data:", error);
        return false;
      }
    };

    const success = extractAndExecuteGraph(responseText);

    if (!success) {
      // If no graph data found, show a message
      chartRef.current.innerHTML =
        '<div style="padding: 20px; text-align: center; color: #666;">No graph data found in response</div>';
    }

    return () => {
      if (chartRef.current && typeof Plotly !== "undefined") {
        Plotly.purge(chartRef.current);
      }
    };
  }, [responseText]);

  return (
    <div className="w-full">
      <div
        ref={chartRef}
        style={{
          height: "500px",
          width: "100%",
          border: "1px solid #e2e8f0",
          borderRadius: "8px",
          backgroundColor: "#fff",
        }}
      />
    </div>
  );
};

export default GraphResponseComponent;
