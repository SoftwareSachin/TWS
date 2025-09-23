from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from app.be_core.logger import logger


class DatasetNameConverter:
    """
    Handles automatic conversion of dataset names (like 'ds1', 'ds2') to their UUID equivalents.
    This provides a functional layer to replace prompt-based conversion with reliable programmatic conversion.
    """

    def __init__(self, tools_map: Optional[Dict[UUID, Dict[str, Any]]] = None):
        """
        Initialize the converter with tools map containing dataset name mappings.

        Args:
            tools_map: Dictionary mapping tool IDs to their dataset information
        """
        self.name_to_uuid_map: Dict[str, str] = {}
        self.uuid_to_name_map: Dict[str, str] = {}
        self._build_conversion_maps(tools_map)

    def _build_conversion_maps(
        self, tools_map: Optional[Dict[UUID, Dict[str, Any]]] = None
    ):
        """Build bidirectional conversion maps from the tools map."""
        if not tools_map:
            return

        for tool_info in tools_map.values():
            if not isinstance(tool_info, dict):
                continue

            dataset_names = tool_info.get("dataset_names", {})

            # dataset_names is a dict: {uuid: name}
            for uuid_str, name in dataset_names.items():
                if name and not name.startswith("Unknown-"):
                    # Build name -> UUID mapping (ds1 -> uuid)
                    self.name_to_uuid_map[name] = uuid_str
                    # Build UUID -> name mapping (uuid -> ds1)
                    self.uuid_to_name_map[uuid_str] = name

    def convert_dataset_identifiers(
        self, dataset_ids: Union[List[Any], Any]
    ) -> List[str]:
        """
        Convert a list of dataset identifiers (names or UUIDs) to standardized UUID strings.

        Args:
            dataset_ids: List of dataset identifiers (can contain names like 'ds1', UUIDs, etc.)

        Returns:
            List of UUID strings with names converted to their corresponding UUIDs
        """
        if not dataset_ids:
            return []

        # Ensure we have a list
        if not isinstance(dataset_ids, list):
            dataset_ids = [dataset_ids]

        converted_ids = []

        for identifier in dataset_ids:
            identifier_str = str(identifier).strip()

            # If it's already a UUID (either as string or UUID object), keep it
            if self._is_uuid_format(identifier_str):
                converted_ids.append(identifier_str)
            # If it's a dataset name (like 'ds1'), convert to UUID
            elif identifier_str in self.name_to_uuid_map:
                uuid_str = self.name_to_uuid_map[identifier_str]
                converted_ids.append(uuid_str)
                logger.info(
                    f"Converted dataset name '{identifier_str}' to UUID '{uuid_str}'"
                )
            else:
                # Unknown identifier - could be a UUID we don't recognize or invalid name
                logger.warning(
                    f"Unknown dataset identifier: '{identifier_str}' - keeping as-is"
                )
                converted_ids.append(identifier_str)

        # Remove duplicates while preserving order
        seen = set()
        unique_converted = []
        for uuid_str in converted_ids:
            if uuid_str not in seen:
                seen.add(uuid_str)
                unique_converted.append(uuid_str)

        return unique_converted

    def _is_uuid_format(self, identifier: str) -> bool:
        """Check if a string is in UUID format."""
        try:
            UUID(identifier)
            return True
        except (ValueError, TypeError):
            return False

    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about available conversions for debugging."""
        return {
            "name_to_uuid_mappings": self.name_to_uuid_map,
            "uuid_to_name_mappings": self.uuid_to_name_map,
            "available_dataset_names": list(self.name_to_uuid_map.keys()),
            "available_uuids": list(self.uuid_to_name_map.keys()),
        }
