export interface IGraphCreate {
  status: string;
  error_message?: string;
  entity_types?: string[];
}

export interface IGraphRead {
  id: string;
  dataset_id: string;
  entities_status: "pending" | "success" | "failed" | "not_started";
  relationships_status: "pending" | "success" | "failed" | "not_started";
  error_message?: string;
  entity_types?: string[];
  created_at: string;
  updated_at: string;
}

export interface IGraphUpdate {
  status?: string;
  error_message?: string;
  entity_types?: string[];
}

export interface IEntityType {
  entity_type: string;
  count: number;
}

export interface IEntityTypesResponse {
  entity_types: IEntityType[];
}

export interface IEntity {
  id: string;
  name: string;
  type: string;
  description: string;
  checked?: boolean;
  instances?: number;
}

export interface IRelationship {
  source_entity: string;
  target_entity: string;
  relationship_type: string;
  relationship_description: string;
}

export interface IGraphEntitiesRelationships {
  entities: IEntity[];
  relationships: IRelationship[];
  total_entities: number;
  total_relationships: number;
}

export interface IEntityExtractionPayload {
  entity_types?: string[];
}

export interface IEntityDeletionPayload {
  types: string[];
}
