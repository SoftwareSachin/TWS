from typing import Optional

from pydantic import BaseModel


class MFAValidationRequest(BaseModel):
    secondfactorauthenticationtoken: Optional[str] = None
    authenticatorcode: str
    access_token: Optional[str] = None


class QRCode(BaseModel):
    QRcode: str


class VerifyAuthCode(BaseModel):
    authCode: str


class TwoFactorSchema(BaseModel):
    TwoFactAuthenticationToken: str
