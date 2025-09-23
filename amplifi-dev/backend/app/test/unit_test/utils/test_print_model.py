from fastapi.encoders import jsonable_encoder
from sqlmodel import Field, SQLModel

from app.utils.print_model import print_model


# Define a sample SQLModel for testing
class Item(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    description: str = None


def test_print_model(capfd):
    item = Item(id=1, name="Test Item", description="This is a test item.")
    print_model("Item Details:", item)
    captured = capfd.readouterr()
    expected_output = f"Item Details: {jsonable_encoder(item)}\n"
    assert captured.out == expected_output
