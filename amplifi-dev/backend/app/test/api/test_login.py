import pytest

from app.be_core.config import settings


@pytest.mark.asyncio
async def test_login_with_wrong_email(test_client):
    form_data = {
        "username": "wrongemail@admin.com",
        "password": settings.FIRST_SUPERUSER_PASSWORD,
        "securityanswer[Project Name]": "Amplifi",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = await test_client.post(
        "/login/access-token", data=form_data, headers=headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid credentials"


@pytest.mark.asyncio
async def test_login_with_wrong_password(test_client):
    form_data = {
        "username": settings.FIRST_SUPERUSER_EMAIL,
        "password": "wrongpassword",
        "securityanswer[Project Name]": "Amplifi",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = await test_client.post(
        "/login/access-token", data=form_data, headers=headers
    )
    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Invalid credentials"
        or response.json()["detail"] == "Your account is blocked"
    )


@pytest.mark.skip(
    reason="Not implemented yet, currently just hardcoded since Loginradius requires"
)
@pytest.mark.asyncio
async def test_login_with_wrong_securityanswer(test_client):
    form_data = {
        "username": settings.FIRST_SUPERUSER_EMAIL,
        "password": settings.FIRST_SUPERUSER_PASSWORD,
        "securityanswer[Project Name]": "incorrect_security_answer",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = await test_client.post(
        "/login/access-token", data=form_data, headers=headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid credentials"


@pytest.mark.skip(reason="User got deleted or disabled on Loginradius")
@pytest.mark.asyncio
async def test_successful_login(test_client):
    form_data = {
        "username": settings.FIRST_SUPERUSER_EMAIL,
        "password": settings.FIRST_SUPERUSER_PASSWORD,
        "securityanswer[Project Name]": "Amplifi",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = await test_client.post(
        "/login/access-token", data=form_data, headers=headers
    )
    assert response.status_code == 200
    response_data = response.json()
    assert "access_token" in response_data
    assert response_data["token_type"] == "bearer"


## works but not changing password so can test other subsequent tests
# @pytest.mark.asyncio
# async def test_change_password(test_client_admin):
#     change_password_data = {"current_password": "admin", "new_password": "newpassword"}
#     response = await test_client_admin.post("/login/change_password", json=change_password_data)
#     assert response.status_code == 200
#     assert response.json()["message"] == "New password generated"
#     assert "access_token" in response.json()["data"]
#     assert response.json()["data"]["token_type"] == "bearer"
