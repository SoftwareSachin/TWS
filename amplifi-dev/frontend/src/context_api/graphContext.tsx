"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

interface GraphContextType {
  // Graph status states
  currentGraphId: string | null;
  graphEntitiesStatus: string | null;
  graphRelationshipsStatus: string | null;

  // Generation status states
  isGeneratingEntities: boolean;
  isGeneratingRelationships: boolean;
  isFetchingStatus: boolean;

  // Graph data
  graphRelationships: any;

  // Actions
  setCurrentGraphId: (id: string | null) => void;
  setGraphEntitiesStatus: (status: string | null) => void;
  setGraphRelationshipsStatus: (status: string | null) => void;
  setIsGeneratingEntities: (generating: boolean) => void;
  setIsGeneratingRelationships: (generating: boolean) => void;
  setIsFetchingStatus: (fetching: boolean) => void;
  setGraphRelationships: (relationships: any) => void;

  // Combined actions
  updateGraphStatus: (
    graphId: string | null,
    entitiesStatus: string | null,
    relationshipsStatus: string | null,
  ) => void;
  resetGraphState: () => void;
}

const GraphContext = createContext<GraphContextType | undefined>(undefined);

export const GraphProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [currentGraphId, setCurrentGraphId] = useState<string | null>(null);
  const [graphEntitiesStatus, setGraphEntitiesStatus] = useState<string | null>(
    null,
  );
  const [graphRelationshipsStatus, setGraphRelationshipsStatus] = useState<
    string | null
  >(null);
  const [isGeneratingEntities, setIsGeneratingEntities] = useState(false);
  const [isGeneratingRelationships, setIsGeneratingRelationships] =
    useState(false);
  const [isFetchingStatus, setIsFetchingStatus] = useState(false);
  const [graphRelationships, setGraphRelationships] = useState(null);

  const updateGraphStatus = (
    graphId: string | null,
    entitiesStatus: string | null,
    relationshipsStatus: string | null,
  ) => {
    setCurrentGraphId(graphId);
    setGraphEntitiesStatus(entitiesStatus);
    setGraphRelationshipsStatus(relationshipsStatus);
  };

  const resetGraphState = () => {
    setCurrentGraphId(null);
    setGraphEntitiesStatus(null);
    setGraphRelationshipsStatus(null);
    setIsGeneratingEntities(false);
    setIsGeneratingRelationships(false);
    setIsFetchingStatus(false);
    setGraphRelationships(null);
  };

  const value: GraphContextType = {
    // States
    currentGraphId,
    graphEntitiesStatus,
    graphRelationshipsStatus,
    isGeneratingEntities,
    isGeneratingRelationships,
    isFetchingStatus,
    graphRelationships,

    // Actions
    setCurrentGraphId,
    setGraphEntitiesStatus,
    setGraphRelationshipsStatus,
    setIsGeneratingEntities,
    setIsGeneratingRelationships,
    setIsFetchingStatus,
    setGraphRelationships,

    // Combined actions
    updateGraphStatus,
    resetGraphState,
  };

  return (
    <GraphContext.Provider value={value}>{children}</GraphContext.Provider>
  );
};

export const useGraph = (): GraphContextType => {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error("useGraph must be used within a GraphProvider");
  }
  return context;
};
