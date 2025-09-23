from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi_pagination import Params
from sqlmodel import and_, col, or_, select, text

from app import crud
from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.be_core.security import (
    check_env,
    generate_temp_password,
    get_loginradius_client,
    get_user_status,
    organization_exists,
    user_exists,
)
from app.deps import user_deps
from app.models import User
from app.models.role_model import Role
from app.schemas.passwordResetSchema import PasswordReqFetch
from app.schemas.response_schema import (
    IDeleteResponseBase,
    IGetResponseBase,
    IGetResponsePaginated,
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import (
    IUserCreate,
    IUserInvite,
    IUserRead,
    IUserReadWithoutGroups,
    IUserReadWithWorkspaces,
    IUserStatus,
    UserData,
    UserDataWithWorkspace,
)
from app.utils.exceptions import (
    UserSelfDeleteException,
)

router = APIRouter()
loginradius = get_loginradius_client()


@router.get("/list")
async def read_users_list(
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
    organization_id: UUID = Query(None, description="Optional organization ID"),
) -> IGetResponsePaginated[IUserReadWithoutGroups]:
    """
    Retrieves a list of users.

    Required roles:
    - admin

    Note:
    - If organisation id is given then it will only fetch user's in that org .
    - Else it will fetch all the users.
    """
    if organization_id:
        logger.info(f"fetching users within organization {organization_id}")
        users = await crud.user.get_users(
            organization_id=organization_id, params=params
        )
        return create_response(data=users)

    else:
        logger.info("fetching all users")
        users = await crud.user.get_multi_paginated(params=params)
        return create_response(data=users)


@router.get("/list/by_role_name")
async def read_users_list_by_role_name(
    name: str = "",
    user_status: Annotated[
        IUserStatus,
        Query(
            title="User status",
            description="User status, It is optional. Default is active",
        ),
    ] = IUserStatus.active,
    role_name: str = "",
    params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponsePaginated[IUserReadWithoutGroups]:
    """
    Retrieves users by role name and status.

    Required roles:
    - admin
    - member
    - developer
    """
    user_status = True if user_status == IUserStatus.active else False
    query = (
        select(User)
        .join(Role, User.role_id == Role.id)
        .where(
            and_(
                col(Role.name).ilike(f"%{role_name}%"),
                User.is_active == user_status,
                or_(
                    col(User.first_name).ilike(f"%{name}%"),
                    col(User.last_name).ilike(f"%{name}%"),
                    text(
                        f"""'{name}' % concat("User".last_name, ' ', "User".first_name)"""
                    ),
                    text(
                        f"""'{name}' % concat("User".first_name, ' ', "User".last_name)"""
                    ),
                ),
            )
        )
        .order_by(User.first_name)
    )
    users = await crud.user.get_multi_paginated(query=query, params=params)
    return create_response(data=users)


@router.get("/{user_id}")
async def get_user_by_id(
    user: User = Depends(user_deps.is_valid_user),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponseBase[IUserReadWithWorkspaces]:
    """
    Retrieves a user by their ID.

    Required roles:
    - admin
    - member
    - developer
    """

    workspace_ids = await crud.user.get_workspace_ids_for_user(user_id=user.id)
    user_data = IUserReadWithWorkspaces(
        **user.dict(exclude={"hashed_password"}, exclude_defaults=True),
        is_active=current_user.is_active,
        workspace_ids=workspace_ids,
        role=user.role.name,
    )
    return create_response(data=user_data)


@router.get("")
async def get_my_data(
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
) -> IGetResponseBase[UserDataWithWorkspace]:
    """
    Gets my user profile information

    Required roles:
    - admin
    - member
    - developer
    """
    workspace_ids = await crud.user.get_workspace_ids_for_user(user_id=current_user.id)
    user_data = UserDataWithWorkspace(**current_user.dict(), workspace_id=workspace_ids)
    return create_response(data=user_data)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_user(
    new_user: IUserCreate = Depends(user_deps.user_exists),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IPostResponseBase[IUserRead]:
    """
    Creates a new user

    Required roles:
    - admin
    """
    user = await crud.user.create_with_role(obj_in=new_user)
    return create_response(data=user)


@router.post("/invite-user")
async def invite_user(
    new_user: IUserInvite,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IPostResponseBase[IUserRead]:
    """
    Invites a new user to the organization.

    Required roles:
    - admin
    """
    # before everything checking wether the organization is a valid organization or not

    organization_check = await organization_exists(new_user=new_user)
    if organization_check:
        return organization_check

    user_check = await user_exists(new_user=new_user)
    if user_check:
        return user_check

    try:
        timeDifference = "10"
        getLRserverTime = True
        sott_data = loginradius.get_sott(
            timeDifference=timeDifference, getLRserverTime=getLRserverTime
        )
    except Exception as e:
        logger.error(f"failed to invite user {e}")
        return {"error": "Failed to generate token."}

    try:
        temp_password = generate_temp_password()
        _register_user(
            sott_data=sott_data,
            new_user=new_user,
            temp_password=temp_password,
        )
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        return {"error": "Failed to register user with LoginRadius."}

    try:
        info = loginradius.account.get_account_profile_by_email(new_user.email)
    except Exception as e:
        logger.error(f"Error retrieving user profile: {e}")
        return {"error": "Failed to retrieve user profile after registration."}

    if info["Uid"]:
        user_id = info["Uid"]
        role = new_user.role
        logger.info(f"{new_user.first_name} registered as {role}")
        try:
            _assign_roles_to_user(user_id, role)
        except Exception as e:
            logger.error(f"Role assignment failed for user {user_id}: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    info = loginradius.account.get_account_profile_by_email(new_user.email)
    role_id = await crud.role.get_roleid_by_name(role=new_user.role)

    new_user_data = _create_user(
        new_user=new_user,
        role_id=role_id,
        temp_password=temp_password,
        status=get_user_status(info),
    )
    new_user = IUserCreate(**new_user_data)
    user = await crud.user.create_with_role(obj_in=new_user)
    return create_response(data=user, message="Invite sent successfully")


@router.delete("/{user_id}")
async def remove_user(
    user_id: UUID = Depends(user_deps.is_valid_user_id),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
) -> IDeleteResponseBase[IUserRead]:
    """
    Deletes a user by thier id

    Required roles:
    - admin
    """
    if current_user.id == user_id:
        raise UserSelfDeleteException()

    user = await crud.user.remove(id=user_id)
    return create_response(data=user, message="User removed")


def _register_user(sott_data, new_user, temp_password):
    user_profile = {
        "email": [{"type": "Primary", "value": new_user.email}],
        "firstName": new_user.first_name,
        "lastName": new_user.last_name,
        "password": temp_password,
        "CustomFields": {
            "clientId": str(new_user.organization_id),
            "tempPassword": temp_password,
        },
    }
    loginradius.authentication.user_registration_by_email(
        auth_user_registration_model=user_profile,
        sott=sott_data,
        email_template="verification-amplifi",
        verification_url=settings.VERIFY_EMAIL_URL,
    )


def _assign_roles_to_user(uid, role):
    account_roles_model = {"roles": [role]}
    try:
        loginradius.role.assign_roles_by_uid(account_roles_model, uid)
    except Exception as e:
        logger.error(f"Error assigning {role} role to Uid:{uid} : {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


def _create_user(new_user, role_id, status, temp_password):
    new_user_data = {
        "first_name": new_user.first_name,
        "last_name": new_user.last_name,
        "email": new_user.email,
        "is_active": True,
        "is_superuser": False,
        "birthdate": datetime.now(),
        "role_id": role_id,
        "phone": "string",
        "state": status,
        "country": "string",
        "address": "string",
        "organization_id": new_user.organization_id,
        "password": temp_password,
    }
    return new_user_data


@router.post(
    "/resend-invite",
    dependencies=[Depends(check_env)],
    include_in_schema=settings.DEPLOYED_ENV in {"local", "azure_dev", "azure_prod"},
)
async def resend_invite(
    email: PasswordReqFetch = Body(...),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin])
    ),
):

    user = await crud.user.get_by_email(email=email.email)
    if not user:
        raise HTTPException(
            status_code=404, detail="Email is not Registered. Please contact Admin"
        )
    elif user.state != "Invited":
        raise HTTPException(
            status_code=400, detail="Access Denied. Please contact Admin"
        )
    logger.info(f"Preparing to send reinvite email to: {email.email}")

    response = loginradius.authentication.auth_resend_email_verification(
        email=email.email, email_template="verification-amplifi"
    )

    if response.get("ErrorCode") == 1122:
        raise HTTPException(
            status_code=429,
            detail="You have reached the email sending limit. Please try again later.",
        )

    if response.get("ErrorCode") == 1025:
        raise HTTPException(
            status_code=429,
            detail="This email address has already been verified, try changing your password.",
        )

    return {"message": "Invite link sent."}
