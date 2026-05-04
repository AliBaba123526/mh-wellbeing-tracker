import { getToken, clearToken } from "./auth.js";

async function request(path, { method = "GET", body = null } = {}) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
  });

  const data = await res.json().catch(() => ({}));

  // IMPORTANT: throw errors instead of returning undefined
  if (res.status === 401) {
    // if it's a protected endpoint, clear token
    if (path !== "/api/auth/login" && path !== "/api/auth/signup") {
      clearToken();
    }
    throw new Error(data?.error || "Unauthorized (401)");
  }

  if (!res.ok) {
    throw new Error(data?.error || `Request failed (${res.status})`);
  }

  return data;
}

export const api = {
  login: (payload) => request("/api/auth/login", { method: "POST", body: payload }),
  signup: (payload) => request("/api/auth/signup", { method: "POST", body: payload }),

  moodCreate: (payload) => request("/api/mood", { method: "POST", body: payload }),
  moodList: () => request("/api/mood"),

  sleepCreate: (payload) => request("/api/sleep", { method: "POST", body: payload }),
  sleepList: () => request("/api/sleep"),

  journalCreate: (payload) => request("/api/journal", { method: "POST", body: payload }),
  journalList: () => request("/api/journal"),

  activityCreate: (payload) => request("/api/activity", { method: "POST", body: payload }),
  activityList: () => request("/api/activity"),

  recList: () => request("/api/recommendations"),
  recAccept: (recId) => request(`/api/recommendations/${recId}/accept`, { method: "PATCH" }),

  alertList: () => request("/api/alerts"),
  alertAck: (alertId) => request(`/api/alerts/${alertId}/ack`, { method: "PATCH" }),

  modelRunsList: () => request("/api/modelruns"),
  runInsights: () => request("/api/ml/run", { method: "POST" }),
};

