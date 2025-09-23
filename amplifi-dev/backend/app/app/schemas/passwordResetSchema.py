from pydantic import BaseModel


class PasswordData(BaseModel):
    oldPassword: str
    newPassword: str


class PasswordReqFetch(BaseModel):
    email: str


class Vtoken(BaseModel):
    vtoken: str


class changePassword(BaseModel):
    vtoken: str
    password: str
