/**
 * The `decodeToken` function decodes a JWT token using the `jsonwebtoken` library in JavaScript.
 * @param token - The `decodeToken` function you provided takes a JWT token as input and attempts to
 * decode it using the `jwt.decode` method from the `jsonwebtoken` library. If decoding is successful,
 * it returns the decoded token payload. If an error occurs during decoding, it logs the error to the
 * console and
 * @returns The `decodeToken` function returns the decoded content of the token if decoding is
 * successful. If an error occurs during decoding, it logs an error message and returns `null`.
 */
import jwt from "jsonwebtoken";

export const decodeToken = (token) => {
  try {
    const decoded = jwt.decode(token);
    return decoded;
  } catch (error) {
    console.error("Error decoding token:", error);
    return null;
  }
};
