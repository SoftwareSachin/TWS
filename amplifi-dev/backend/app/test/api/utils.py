empty_get_response = {
    "message": "Data paginated correctly",
    "meta": {},
    "data": {
        "items": [],
        "total": 0,
        "page": 1,
        "size": 50,
        "pages": 0,
        "previous_page": None,
        "next_page": None,
    },
}


def is_empty_response(response) -> None:
    assert response.status_code == 200
    assert response.json() == empty_get_response


def is_invalid_uuid_response(response) -> None:
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "uuid_parsing"
    assert response.json()["detail"][0]["input"] == "not_valid_uuid"
