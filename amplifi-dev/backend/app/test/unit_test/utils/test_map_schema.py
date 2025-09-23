from pydantic import BaseModel

from app.utils.map_schema import map_models_schema


class TestItemSchema(BaseModel):
    id: int
    name: str
    description: str = None


def test_map_models_schema():
    items = [
        TestItemSchema(id=1, name="Test Item 1", description="This is a test item 1."),
        TestItemSchema(id=2, name="Test Item 2", description="This is a test item 2."),
    ]
    mapped_schemas = map_models_schema(TestItemSchema, items)
    expected_schemas = [TestItemSchema.model_validate(item) for item in items]
    for expected, actual in zip(expected_schemas, mapped_schemas):
        assert expected.dict() == actual.dict()
