document.addEventListener("DOMContentLoaded", function () {
    const seeMoreLink = document.querySelector(".see-more-link");
    // Only rendered when there are more posts than posts_to_show
    if (!seeMoreLink) return;

    const hiddenPosts = document.querySelectorAll(".post.hidden");

    seeMoreLink.addEventListener("click", function (e) {
        e.preventDefault();

        hiddenPosts.forEach(post => {
            post.classList.remove("hidden");
        });

        // Remove the whole <li>, not just the link: the empty item would
        // otherwise keep its cell in the posts grid and leave a gap
        seeMoreLink.closest(".see-more-item").remove();
    });
});

// Header: copy the email address. It is assembled here rather than written in
// the HTML so scrapers can't read it; falls back to mailto if copying fails.
document.addEventListener("DOMContentLoaded", function () {
    const button = document.querySelector("[data-copy-email]");
    if (!button) return;

    const label = button.querySelector("span");
    const email = ["miguelmndez", "gmail.com"].join("@");

    button.addEventListener("click", function () {
        if (!navigator.clipboard) {
            window.location.href = "mailto:" + email;
            return;
        }
        navigator.clipboard.writeText(email).then(function () {
            label.textContent = "Copied!";
            setTimeout(function () { label.textContent = "Copy email"; }, 1600);
        }, function () {
            window.location.href = "mailto:" + email;
        });
    });
});

// Header: make the photo overlay feel like a detector running live
// (frame counter, wobbling FPS/latency/confidence, slightly jittering boxes).
document.addEventListener("DOMContentLoaded", function () {
    const cam = document.querySelector("[data-live-cam]");
    if (!cam || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const frameEl = cam.querySelector("[data-frame]");
    const fpsEl = cam.querySelector("[data-fps]");
    const msEl = cam.querySelector("[data-ms]");
    const confEls = cam.querySelectorAll("[data-conf]");
    const boxes = cam.querySelectorAll(".hero-box");
    const baseConf = [0.98, 0.96];
    let frame = 0;

    // Pause all updates while the photo is scrolled out of view
    let visible = true;
    if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (entries) {
            visible = entries[0].isIntersecting;
        }).observe(cam);
    }

    setInterval(function () {
        if (!visible) return;
        frame++;
        frameEl.textContent = String(frame).padStart(6, "0");
    }, 40); // 25 fps

    setInterval(function () {
        if (!visible) return;
        fpsEl.textContent = (24.6 + Math.random() * 0.8).toFixed(1);
        msEl.textContent = Math.round(15 + Math.random() * 7);
        // Stays below 1.00 so it never reads as a fake-perfect score
        confEls.forEach(function (el, i) {
            el.textContent = (baseConf[i] + Math.random() * 0.014).toFixed(2);
        });
    }, 450);

    // Start jittering only after the boxes' lock-on animation has finished
    setTimeout(function () {
        setInterval(function () {
            if (!visible) return;
            boxes.forEach(function (box) {
                const dx = (Math.random() - 0.5) * 2.4;
                const dy = (Math.random() - 0.5) * 2.4;
                const s = 1 + (Math.random() - 0.5) * 0.012;
                box.style.transform = "translate(" + dx + "px," + dy + "px) scale(" + s + ")";
            });
        }, 160);
    }, 2300);
});

// Header: the visitor's cursor gets tracked as "ID 02". The box follows with a
// little lag (like a smoothed track), leaves a trail, and snaps onto links.
document.addEventListener("DOMContentLoaded", function () {
    const hero = document.querySelector(".hero");
    const box = hero && hero.querySelector(".hero-visitor");
    if (!box || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    // Same condition as the CSS that shows .hero-tracker: desktop with a real mouse
    const enabled = window.matchMedia("(min-width: 768px) and (hover: hover) and (pointer: fine)");

    const label = box.querySelector("[data-visitor-label]");
    const line = hero.querySelector(".hero-trail polyline");
    const SMOOTHING = 0.18;
    const TRAIL_LENGTH = 40;
    let current = null;
    let target = null;
    let trail = [];
    let inside = false;
    let lastMove = 0;
    let confidence = 0.94;
    let hideTimer;
    let running = false;

    setInterval(function () { confidence = 0.91 + Math.random() * 0.06; }, 500);

    function tick() {
        current.x += (target.x - current.x) * SMOOTHING;
        current.y += (target.y - current.y) * SMOOTHING;
        current.w += (target.w - current.w) * SMOOTHING;
        current.h += (target.h - current.h) * SMOOTHING;
        box.style.transform = "translate(" + current.x + "px," + current.y + "px)";
        box.style.width = current.w + "px";
        box.style.height = current.h + "px";

        const cx = current.x + current.w / 2;
        const cy = current.y + current.h / 2;
        const last = trail[trail.length - 1];
        if (inside && (!last || Math.hypot(cx - last[0], cy - last[1]) > 3)) trail.push([cx, cy]);
        // The trail shrinks away once the cursor stops moving
        if (trail.length > TRAIL_LENGTH || (performance.now() - lastMove > 250 && trail.length)) trail.shift();
        line.setAttribute("points", trail.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" "));

        // Stop the loop when there is nothing left to animate
        const settled = Math.abs(target.x - current.x) < 0.5 && Math.abs(target.y - current.y) < 0.5;
        if (inside || trail.length || !settled) requestAnimationFrame(tick);
        else running = false;
    }

    hero.addEventListener("pointermove", function (e) {
        if (e.pointerType !== "mouse" || !enabled.matches) return;
        const rect = hero.getBoundingClientRect();
        const link = e.target.closest(".hero-link");

        if (link) {
            const lr = link.getBoundingClientRect();
            target = { x: lr.left - rect.left - 6, y: lr.top - rect.top - 6, w: lr.width + 12, h: lr.height + 12 };
            label.textContent = "button · " + link.textContent.trim() + " 0.99";
        } else {
            target = { x: e.clientX - rect.left - 32, y: e.clientY - rect.top - 32, w: 64, h: 64 };
            label.textContent = "ID 02 · visitor " + confidence.toFixed(2);
        }

        // A new track starts where the cursor enters, not where the last one ended
        if (!inside) {
            current = Object.assign({}, target);
            trail = [];
        }
        inside = true;
        lastMove = performance.now();
        clearTimeout(hideTimer);
        box.classList.remove("is-lost");
        box.classList.add("is-on");
        if (!running) {
            running = true;
            requestAnimationFrame(tick);
        }
    });

    hero.addEventListener("pointerleave", function () {
        if (!inside) return;
        inside = false;
        box.classList.add("is-lost");
        label.textContent = "ID 02 · lost";
        hideTimer = setTimeout(function () { box.classList.remove("is-on"); }, 900);
    });
});

// Vitae: single-open accordion; the clicked row stays put while others collapse
(function () {
    document.querySelectorAll("[data-vitae]").forEach(function (list) {
        const rows = Array.from(list.querySelectorAll(".vitae-row"));

        function setOpen(row, open) {
            row.classList.toggle("is-open", open);
            row.querySelector(".vitae-toggle").setAttribute("aria-expanded", open);
        }

        // Rows above the clicked one collapse as it opens, which would push it
        // up the screen; compensate scroll during the 350ms transition.
        function anchor(el) {
            const y0 = el.getBoundingClientRect().top;
            const until = performance.now() + 450;
            (function step() {
                const dy = el.getBoundingClientRect().top - y0;
                if (Math.abs(dy) > .5) window.scrollBy({ top: dy, behavior: "instant" });
                if (performance.now() < until) requestAnimationFrame(step);
            })();
        }

        rows.forEach(function (row) {
            const toggle = row.querySelector(".vitae-toggle");
            toggle.addEventListener("click", function () {
                const open = !row.classList.contains("is-open");
                anchor(toggle);
                rows.forEach(function (r) { setOpen(r, r === row && open); });
            });

            // Crosshair follows the pointer over the row header (transform only, no layout)
            const cross = row.querySelector(".vitae-cross");
            toggle.addEventListener("pointermove", function (e) {
                const r = row.getBoundingClientRect();
                cross.style.transform = "translate(" + (e.clientX - r.left) + "px," + (e.clientY - r.top) + "px)";
            });
        });
    });
})();

// News: show the latest entries and an "All news" toggle for the rest. The
// list is only collapsed here, so without JS every entry stays visible.
(function () {
    var list = document.querySelector("[data-news-list]");
    var more = document.querySelector("[data-news-more]");
    if (!list || !more) return;
    var label = more.textContent;
    list.classList.add("is-collapsed");
    more.hidden = false;
    more.addEventListener("click", function () {
        var open = list.classList.toggle("is-collapsed") === false;
        more.setAttribute("aria-expanded", open);
        more.textContent = open ? "Show less ↑" : label;
    });
})();

// Bio: entities show their "TYPE score" tag on hover (CSS), keyboard focus
// and tap. Tags that would overflow the right edge hang leftwards instead.
(function () {
    var root = document.querySelector("[data-bio-ner]");
    if (!root) return;
    var ents = root.querySelectorAll(".ent");

    function fit() {
        var limit = root.getBoundingClientRect().right;
        ents.forEach(function (el) {
            // rough tag width: ~7px per mono character of "TYPE 0.99" plus padding
            var tagW = ((el.dataset.t || "").length + 5) * 7 + 10;
            var first = el.getClientRects()[0];
            el.classList.toggle("is-flip", !!first && first.left + tagW > limit);
        });
    }
    fit();
    var t;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(fit, 150); });

    // Tap toggles a tag on plain-text entities; links keep their normal behaviour
    root.addEventListener("click", function (e) {
        var el = e.target.closest("span.ent");
        ents.forEach(function (x) { if (x !== el) x.classList.remove("is-active"); });
        if (el) el.classList.toggle("is-active");
    });
    document.addEventListener("click", function (e) {
        if (!root.contains(e.target)) ents.forEach(function (x) { x.classList.remove("is-active"); });
    });
})();
