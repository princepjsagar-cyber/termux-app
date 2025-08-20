export default {
  async scheduled(event, env, ctx) {
    const list = (env.KEEPALIVE_URLS || "").split(",").map(u => u.trim()).filter(Boolean);
    if (list.length === 0) {
      return;
    }
    await Promise.all(
      list.map(async (u) => {
        try {
          const res = await fetch(u, { method: "GET", headers: { "User-Agent": "keepalive" } });
          // Drain body to free connection
          await res.arrayBuffer().catch(() => {});
        } catch (e) {
          // ignore
        }
      })
    );
  },
};