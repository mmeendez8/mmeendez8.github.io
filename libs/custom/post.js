// Post page behaviour (_layouts/post.html):
//  - "#" anchors on h2/h3
//  - code block header: language or data-title label + copy button
//  - "On this page" index (sidebar + mobile card) built from the h2s
//  - reading progress bar, title in the sticky bar, and the index's
//    active section with a per-section fill (--p)
document.addEventListener("DOMContentLoaded", function () {
    const root = document.querySelector("[data-post]");
    const prose = root && root.querySelector(".post-prose");
    if (!prose) return;

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
        return lines
            .filter(function (l) { return lang !== "console" || /^\s*\$ /.test(l); })
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
                });
            });
            head.appendChild(btn);
        }
        block.prepend(head);
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
        root.style.setProperty("--read", read.toFixed(4));
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
