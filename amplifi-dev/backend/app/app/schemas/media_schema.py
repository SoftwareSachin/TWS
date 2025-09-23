from uuid import UUID

from app.models.media_model import MediaBase
from app.utils.partial import optional


class IMediaCreate(MediaBase):
    pass


# All these fields are optional
@optional()
class IMediaUpdate(MediaBase):
    pass


class IMediaRead(MediaBase):
    id: UUID | str
    link: str | None = None
