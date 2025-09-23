from enum import Enum
from uuid import UUID

from app.models.role_model import RoleBase
from app.utils.partial import optional


class IRoleCreate(RoleBase):
    pass


# All these fields are optional
@optional()
class IRoleUpdate(RoleBase):
    pass


class IRoleRead(RoleBase):
    id: UUID


class IRoleEnum(str, Enum):
    admin = "Amplifi_Admin"
    member = "Amplifi_Member"
    developer = "Amplifi_Developer"
