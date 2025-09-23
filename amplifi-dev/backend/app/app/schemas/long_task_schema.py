from enum import Enum


class ITaskType(str, Enum):
    ingestion = "ingestion"
    build_graph = "build_graph"
    execute_workflow = "execute_workflow"
    pull_files = "pull_files"
    train_status = "train_status"
