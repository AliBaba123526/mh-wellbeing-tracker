export function toast(message, type = "ok") {
  const el = document.createElement("div");
  el.className = `toast ${type === "ok" ? "ok" : "err"}`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

export function fmtDate(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString();
}
