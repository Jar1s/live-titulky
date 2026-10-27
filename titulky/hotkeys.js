/* Globálne klávesové skratky + "?" pomocník. Zdieľané všetkými stránkami.
   Stránka si pridá vlastné: HK.add("k", "Kamera", fn). */
(function () {
  const NAV = [
    ["o", "Overlay", "/"], ["s", "Studio", "/studio"],
    ["f", "Flow", "/flow"], ["l", "Lab", "/lab"],
    ["z", "Záznamy", "/sessions"], ["n", "Nastavenia", "/settings"],
  ];
  const extra = [];   // [{key, desc, fn}]
  let helpEl = null;

  function typing(e) {
    const t = e.target;
    return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
                 t.tagName === "SELECT" || t.isContentEditable);
  }

  function buildHelp() {
    helpEl = document.createElement("div");
    helpEl.id = "hkHelp";
    helpEl.style.cssText =
      "position:fixed;inset:0;z-index:999;display:none;align-items:center;" +
      "justify-content:center;background:rgba(10,11,13,.62);" +
      "backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)";
    const card = document.createElement("div");
    card.style.cssText =
      "background:linear-gradient(180deg,#2b2e34,#24262b);" +
      "border:1px solid rgba(255,255,255,.08);border-radius:24px;" +
      "box-shadow:0 18px 44px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.09);" +
      "padding:26px 30px;min-width:340px;max-width:460px;color:#f5f5f7;" +
      "font:14px/1.5 -apple-system,'SF Pro Text',Inter,sans-serif";
    let rows = "";
    const kbd = k =>
      `<kbd style="background:#1d1f24;border:1px solid rgba(255,255,255,.1);` +
      `border-radius:7px;padding:2px 9px;font:600 12px ui-monospace,monospace;` +
      `min-width:26px;display:inline-block;text-align:center">${k}</kbd>`;
    const row = (k, d) =>
      `<div style="display:flex;justify-content:space-between;align-items:center;` +
      `gap:26px;padding:5px 0">${kbd(k)}<span style="color:#a6adb8">${d}</span></div>`;
    if (extra.length) {
      rows += `<div style="font:700 10.5px sans-serif;letter-spacing:.1em;` +
        `text-transform:uppercase;color:#6b7280;margin:2px 0 6px">Táto stránka</div>`;
      extra.forEach(x => rows += row(x.key === " " ? "Space" : x.key.toUpperCase(), x.desc));
    }
    rows += `<div style="font:700 10.5px sans-serif;letter-spacing:.1em;` +
      `text-transform:uppercase;color:#6b7280;margin:14px 0 6px">Navigácia</div>`;
    NAV.forEach(([k, d]) => rows += row(k.toUpperCase(), d));
    rows += row("?", "tento pomocník") + row("Esc", "zavrieť");
    card.innerHTML =
      `<div style="font:800 17px -apple-system,'SF Pro Display',sans-serif;` +
      `margin-bottom:12px">Klávesové skratky</div>` + rows;
    helpEl.appendChild(card);
    helpEl.onclick = ev => { if (ev.target === helpEl) toggleHelp(false); };
    document.body.appendChild(helpEl);
  }

  function toggleHelp(on) {
    if (!helpEl) buildHelp();
    const show = on !== undefined ? on : helpEl.style.display === "none";
    helpEl.style.display = show ? "flex" : "none";
  }

  document.addEventListener("keydown", e => {
    if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "?") { e.preventDefault(); toggleHelp(); return; }
    if (e.key === "Escape" && helpEl && helpEl.style.display !== "none") {
      toggleHelp(false); return;
    }
    if (helpEl && helpEl.style.display === "flex") return;
    for (const x of extra) {
      if (x.fn && e.key.toLowerCase() === x.key) {
        e.preventDefault(); x.fn(); return;
      }
    }
    for (const [k, , url] of NAV) {
      if (e.key.toLowerCase() === k &&
          !(window.HK_SKIP_NAV || []).includes(k)) {
        location.href = url; return;
      }
    }
  });

  window.HK = {
    add(key, desc, fn) { extra.push({key: key.toLowerCase(), desc, fn}); },
    doc(key, desc) { extra.push({key: key.toLowerCase(), desc, fn: null}); },
    help: toggleHelp,
  };
})();
