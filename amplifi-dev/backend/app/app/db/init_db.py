from sqlmodel.ext.asyncio.session import AsyncSession

from app import crud
from app.be_core.config import settings
from app.models.organization_model import Organization
from app.models.role_model import Role
from app.schemas.group_schema import IGroupCreate
from app.schemas.hero_schema import IHeroCreate
from app.schemas.platform_schema import IDeploymentInfoCreate
from app.schemas.role_schema import IRoleCreate
from app.schemas.team_schema import ITeamCreate
from app.schemas.user_schema import IUserCreate

roles: list[IRoleCreate] = [
    IRoleCreate(name="Amplifi Admin", description="Amplifi Admin role"),
    IRoleCreate(name="Amplifi Member", description="Amplifi Member role"),
    IRoleCreate(name="Amplifi Developer", description="Amplifi Developer role"),
    IRoleCreate(name="Amplifi_Admin", description="Amplifi Admin role"),
    IRoleCreate(name="Amplifi_Member", description="Amplifi Member role"),
    IRoleCreate(name="Amplifi_Developer", description="Amplifi Developer role"),
]


groups: list[IGroupCreate] = [
    IGroupCreate(name="GR1", description="This is the first group")
]

users: list[dict[str, str | IUserCreate]] = [
    {
        "data": IUserCreate(
            first_name="Admin",
            last_name="FastAPI",
            password=settings.FIRST_SUPERUSER_PASSWORD,
            email="admin@admin.com",
            is_superuser=False,
        ),
        "role": "Amplifi Admin",
    },
    {
        "data": IUserCreate(
            first_name="Member",
            last_name="FastAPI",
            password=settings.FIRST_SUPERUSER_PASSWORD,
            email="member@example.com",
            is_superuser=False,
        ),
        "role": "Amplifi Member",
    },
    {
        "data": IUserCreate(
            first_name="Developer",
            last_name="FastAPI",
            password=settings.FIRST_SUPERUSER_PASSWORD,
            email="developer@example.com",
            is_superuser=False,
        ),
        "role": "Amplifi Developer",
    },
    {
        "data": IUserCreate(
            first_name="Amplifi",
            last_name="FastAPI ",
            password=settings.FIRST_SUPERUSER_PASSWORD,
            email=settings.FIRST_SUPERUSER_EMAIL,
            is_superuser=True,
        ),
        "role": "Amplifi Admin",
    },
]

if settings.DEPLOYED_ENV == "local":
    users.append(
        {
            "data": IUserCreate(
                first_name="Test",
                last_name="Superuser",
                password=settings.TEST_USER_PASSWORD,
                email=settings.TEST_USER_EMAIL,
                is_superuser=True,
                organization_id=settings.TEST_ORG_UUID,
                state="Active",
            ),
            "role": "Amplifi Admin",
        },
    )
    users.append(
        {
            "data": IUserCreate(
                first_name="Local",
                last_name="Superuser",
                password=settings.LOCAL_USER_PASSWORD,
                email=settings.LOCAL_USER_EMAIL,
                is_superuser=True,
                organization_id=settings.LOCAL_ORG_UUID,
                state="Active",
            ),
            "role": "Amplifi Admin",
        },
    )
    roles.append(
        Role(
            name="TestRole",
            description="Role for test users, no perms",
            id=settings.TEST_ROLE_ID,
        )
    )
    roles.append(
        Role(
            name="TestRole",
            description="Role for local users, no perms",
            id=settings.LOCAL_ROLE_ID,
        )
    )
teams: list[ITeamCreate] = [
    ITeamCreate(name="Preventers", headquarters="Sharp Tower"),
    ITeamCreate(name="Z-Force", headquarters="Sister Margaret's Bar"),
]

# Ignore [B106:hardcoded_password_funcarg] Possible hardcoded password
heroes: list[dict[str, str | IHeroCreate]] = [
    {
        "data": IHeroCreate(
            name="Deadpond", secret_name="Dive Wilson", age=21
        ),  # nosec
        "team": "Z-Force",
    },
    {
        "data": IHeroCreate(
            name="Rusty-Man", secret_name="Tommy Sharp", age=48
        ),  # nosec
        "team": "Preventers",
    },
]


async def _init_test_users(db_session: AsyncSession):
    for user in users:
        current_user = await crud.user.get_by_email(
            email=user["data"].email, db_session=db_session
        )
        role = await crud.role.get_role_by_name(
            name=user["role"], db_session=db_session
        )
        if not current_user:
            user["data"].role_id = role.id
            await crud.user.create_with_role(obj_in=user["data"], db_session=db_session)


async def _init_groups(db_session: AsyncSession):
    for group in groups:
        current_group = await crud.group.get_group_by_name(
            name=group.name, db_session=db_session
        )
        if not current_group:
            current_user = await crud.user.get_by_email(
                email=users[0]["data"].email, db_session=db_session
            )
            new_group = await crud.group.create(
                obj_in=group, created_by_id=current_user.id, db_session=db_session
            )
            current_users = []
            for user in users:
                current_users.append(
                    await crud.user.get_by_email(
                        email=user["data"].email, db_session=db_session
                    )
                )
            await crud.group.add_users_to_group(
                users=current_users, group_id=new_group.id, db_session=db_session
            )


async def _init_roles(db_session: AsyncSession):
    for role in roles:
        role_current = await crud.role.get_role_by_name(
            name=role.name, db_session=db_session
        )
        if not role_current:
            await crud.role.create(obj_in=role, db_session=db_session)


async def init_db(db_session: AsyncSession) -> None:
    await _init_roles(db_session)
    if settings.DEPLOYED_ENV == "local":
        org = await crud.organization.get_organization_by_id(
            organization_id=settings.TEST_ORG_UUID, db_session=db_session
        )
        if not org:
            organization = Organization(
                name="testorg", domain="example.com", id=settings.TEST_ORG_UUID
            )
            db_session.add(organization)
            await db_session.commit()
            await db_session.refresh(organization)

    await _init_test_users(db_session)

    # await _init_groups(db_session)

    # Hardcoded Deployment Data
    deployment_version = "1.0.0"
    product_docs = "https://example.com/product-docs"
    technical_docs = "https://example.com/technical-docs"

    current_deployment = await crud.deployment_info.get_by_version(
        version=deployment_version, db_session=db_session
    )
    if not current_deployment:
        await crud.deployment_info.create(
            obj_in=IDeploymentInfoCreate(
                version=deployment_version,
                product_documentation_link=product_docs,
                technical_documentation_link=technical_docs,
            ),
            db_session=db_session,
        )
