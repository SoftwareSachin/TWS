from typing import Optional

from pydantic import BaseModel

from .user_schema import IUserRead


class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str
    user: IUserRead


class AccessTokenRead(BaseModel):
    access_token: str
    token_type: str


class TokenRead(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str
    jwt_token: Optional[str] = None
    secondFactorAuthenticationToken: Optional[str] = None
    QrCode: Optional[str] = None
    isMfaEnabled: Optional[bool] = None
    email: Optional[str] = None
    FirstLogin: Optional[bool] = True


class RefreshToken(BaseModel):
    refresh_token: str
