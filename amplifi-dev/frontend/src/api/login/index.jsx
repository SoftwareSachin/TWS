import { http } from "..";

export const loginApi = async (formData) => {
  formData["username"] = formData.username.toLowerCase();
  const res = await http.post(`/login/access-token`, formData, {
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  });
  return res.data;
};

export const UserList = async () => {
  return await http.get(`/user/list?page=1&size=50`);
};

export const validateAuthCode = (authToken, authCode, accessToken) => {
  const payload = {
    authenticatorcode: authCode,
    access_token: accessToken || "",
    secondfactorauthenticationtoken: authToken,
  }; // JSON body
  return http.post("/login/validate-code", payload);
};

export const logout = () => {
  return http.get(`/logout`);
};

export const checkForFirstTime = (body) => {
  return http.post(`/is_reset_password_required`, body);
};

export const resetPassword = (body, token) => {
  return http.post(`/reset_password`, body, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
};

export const verifyMailId = (body) => {
  return http.post(`/login/verify-email`, body);
};

export const forgotPasswordInvite = (body) => {
  return http.post(`/login/forgot-password`, body);
};

export const forgotPassword = (body) => {
  return http.post(`/login/change-password`, body);
};
