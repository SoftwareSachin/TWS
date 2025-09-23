"use client";

import React from "react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const PlotlyGraphComponent = ({ chartData: { data, layout } }) => {
  return (
    <div className="">
      <Plot
        className=""
        data={data}
        layout={layout}
        //config={chartData.config}
      />
    </div>
  );
};

export default PlotlyGraphComponent;
