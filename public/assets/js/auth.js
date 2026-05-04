const TOKEN_KEY = "mh_token";

export function setToken(token) { localStorage.setItem(TOKEN_KEY, token); }
export function getToken() { return localStorage.getItem(TOKEN_KEY); }
export function clearToken() { localStorage.removeItem(TOKEN_KEY); }

export function requireLogin() {
  const t = getToken();
  if (!t) window.location.href = "/login.html";
  return t;
}
