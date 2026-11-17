// Post page behaviour (_layouts/post.html):
//  - "#" anchors on h2/h3
//  - code block header: language or data-title label + copy button
//  - tables: scroll wrapper, number / yes-no column alignment, ✅/❌ marks
//  - "On this page" index (sidebar + mobile card) built from the h2s
//  - reading progress bar, title in the sticky bar, and the index's
//    active section with a per-section fill (--p)
document.addEventListener("DOMContentLoaded", function () {
    const root = document.querySelector("[data-post]");
    const prose = root && root.querySelector(".post-prose");
    if (!prose) return;
    const progress = root.querySelector(".post-progress");

    // ---- heading anchors
    prose.querySelectorAll("h2[id], h3[id]").forEach(function (h) {
        const a = document.createElement("a");
        a.className = "post-anchor";
        a.href = "#" + h.id;
        a.textContent = "#";
        a.setAttribute("aria-label", "Link to this section");
        h.prepend(a);
    });

    // ---- code blocks
    const SHELL = ["bash", "sh", "shell", "zsh", "console", "shell-session"];
    const HIDDEN_LANGS = ["plaintext", "text", "plain"];

    // Copy what the reader would type: drop "$ " prompts and, in console
    // blocks that have prompts, the output lines.
    function copyText(code, lang) {
        let text = code.innerText.replace(/\n$/, "");
        if (SHELL.indexOf(lang) === -1) return text;
        const lines = text.split("\n");
        const hasPrompt = lines.some(function (l) { return /^\s*\$ /.test(l); });
        if (!hasPrompt) return text;
        let continued = false;
        return lines
            .filter(function (l) {
                const keep = lang !== "console" || /^\s*\$ /.test(l) || continued;
                continued = keep && /\\\s*$/.test(l);
                return keep;
            })
            .map(function (l) { return l.replace(/^(\s*)\$ /, "$1"); })
            .join("\n");
    }

    const blocks = prose.querySelectorAll("div.highlighter-rouge, .post-prose > pre");
    blocks.forEach(function (block) {
        const code = block.querySelector("code");
        if (!code) return;
        const lang = ((block.className.match(/language-([\w-]+)/) || [])[1] || "").toLowerCase();
        const title = block.getAttribute("data-title");
        const label = title || (HIDDEN_LANGS.indexOf(lang) === -1 ? lang : "");

        const head = document.createElement("div");
        head.className = "code-head" + (label ? "" : " is-bare");
        if (label) {
            const span = document.createElement("span");
            span.textContent = label;
            head.appendChild(span);
        }
        if (navigator.clipboard) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "code-copy";
            btn.textContent = "Copy";
            btn.setAttribute("aria-label", "Copy code to clipboard");
            btn.addEventListener("click", function () {
                navigator.clipboard.writeText(copyText(code, lang)).then(function () {
                    btn.textContent = "Copied";
                    btn.classList.add("is-done");
                    setTimeout(function () {
                        btn.textContent = "Copy";
                        btn.classList.remove("is-done");
                    }, 1400);
                }, function () {
                    btn.textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "Press ⌘C" : "Press Ctrl+C";
                    const range = document.createRange();
                    range.selectNodeContents(code);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(range);
                    setTimeout(function () { btn.textContent = "Copy"; }, 2400);
                });
            });
            head.appendChild(btn);
        }
        block.prepend(head);
    });

    // ---- tables
    const YES = "\u2705", NO = "\u274C";
    const NUM = /^[-+]?[\d.,]+\s*(%|ms|s|x)?(\s*\(x?[\d.,]+\))?$/i;
    const PLACEHOLDER = /^(x|-|\u2014|n\/a)$/i;
    prose.querySelectorAll("table").forEach(function (table) {
        if (!table.parentElement.classList.contains("table-wrapper")) {
            const wrap = document.createElement("div");
            wrap.className = "table-wrapper";
            table.parentNode.insertBefore(wrap, table);
            wrap.appendChild(table);
        }
        const rows = Array.from(table.rows);
        const cols = rows.reduce(function (n, r) { return Math.max(n, r.cells.length); }, 0);
        for (let c = 0; c < cols; c++) {
            const body = rows.slice(1).map(function (r) { return r.cells[c]; }).filter(Boolean);
            const filled = body.filter(function (td) { return td.textContent.trim() !== ""; });
            if (!filled.length) continue;
            const isMark = filled.every(function (td) {
                const t = td.textContent.trim();
                return t.indexOf(YES) === 0 || t.indexOf(NO) === 0;
            });
            // placeholders ("X", "-", "n/a") may sit in a number column
            const values = filled.filter(function (td) { return !PLACEHOLDER.test(td.textContent.trim()); });
            const isNum = !isMark && c > 0 && values.length > 0 &&
                values.every(function (td) { return NUM.test(td.textContent.trim()); });
            const cls = isMark ? "is-mark" : (isNum ? "is-num" : "");
            rows.forEach(function (r) {
                const cell = r.cells[c];
                if (!cell) return;
                if (cls) cell.classList.add(cls);
                if (isMark && cell.tagName === "TD") {
                    const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
                    let node;
                    while ((node = walker.nextNode())) {
                        const i = node.data.search(/\u2705|\u274C/);
                        if (i === -1) continue;
                        const yes = node.data.charAt(i) === YES;
                        const m = document.createElement("span");
                        m.className = "mark " + (yes ? "is-yes" : "is-no");
                        m.textContent = yes ? "\u2713" : "\u2715";
                        m.setAttribute("role", "img");
                        m.setAttribute("aria-label", yes ? "Yes" : "No");
                        const rest = node.splitText(i);
                        rest.data = rest.data.slice(1);
                        rest.parentNode.insertBefore(m, rest);
                        // whatever follows the mark (e.g. "(?)") becomes a muted note
                        if (rest.data.trim()) {
                            const n = document.createElement("span");
                            n.className = "mark-note";
                            n.textContent = rest.data.trim();
                            rest.replaceWith(n);
                        }
                        break;
                    }
                }
            });
        }
        table.querySelectorAll("td").forEach(function (td) {
            const t = td.textContent.trim();
            if (t === "" && !td.children.length) td.classList.add("is-empty");
            else if (PLACEHOLDER.test(t)) td.classList.add("is-na");
        });
    });

    // ---- "On this page"
    const heads = Array.from(prose.querySelectorAll("h2[id]"));
    const toc = root.querySelector(".post-toc");
    const mobile = root.querySelector(".post-toc-mobile");
    const label = function (h) { return h.textContent.replace(/^#/, "").trim(); };
    let links = [];

    if (heads.length >= 3 && toc && mobile) {
        heads.forEach(function (h) {
            [toc, mobile].forEach(function (list, i) {
                const li = document.createElement("li");
                const a = document.createElement("a");
                a.href = "#" + h.id;
                a.textContent = label(h);
                li.appendChild(a);
                list.querySelector("ol").appendChild(li);
                if (i === 0) links.push(a);
            });
        });
        toc.hidden = false;
        mobile.hidden = false;
        mobile.addEventListener("click", function (e) {
            if (e.target.closest("a")) mobile.open = false;
        });
    }

    // ---- scroll state
    let ticking = false;
    function update() {
        const r = prose.getBoundingClientRect();
        const total = r.height - window.innerHeight * 0.6;
        const read = Math.min(1, Math.max(0, -r.top / Math.max(1, total)));
        if (progress) progress.style.setProperty("--read", read.toFixed(4));
        root.classList.toggle("is-scrolled", r.top < 0);
        root.classList.toggle("is-reading", r.top < window.innerHeight * 0.5);

        if (links.length) {
            const line = window.innerHeight * 0.25;
            const tops = heads.map(function (h) { return h.getBoundingClientRect().top; });
            let cur = -1;
            tops.forEach(function (t, i) { if (t <= line) cur = i; });
            links.forEach(function (a, i) {
                const stop = i + 1 < tops.length ? tops[i + 1] : r.bottom;
                const p = Math.min(1, Math.max(0, (line - tops[i]) / Math.max(1, stop - tops[i])));
                a.style.setProperty("--p", i === cur ? p.toFixed(3) : (i < cur ? 1 : 0));
                a.classList.toggle("active", i === cur);
                a.classList.toggle("done", i < cur);
                if (i === cur) a.setAttribute("aria-current", "location");
                else a.removeAttribute("aria-current");
            });
        }
        ticking = false;
    }

    window.addEventListener("scroll", function () {
        if (!ticking) {
            ticking = true;
            window.requestAnimationFrame(update);
        }
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
});
