from uuid import UUID

from app.models.hero_model import HeroBase
from app.models.team_model import TeamBase
from app.utils.partial import optional

from .user_schema import IUserBasicInfo


class ITeamCreate(TeamBase):
    pass


# All these fields are optional
@optional()
class ITeamUpdate(TeamBase):
    pass


class ITeamRead(TeamBase):
    id: UUID
    created_by: IUserBasicInfo


class ITeamReadWithHeroes(ITeamRead):
    heroes: list[HeroBase]
