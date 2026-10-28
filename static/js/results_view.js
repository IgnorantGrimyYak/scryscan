// results_view.js - shared results renderer for the owner page and the
// read-only share pages.
//
// createResultsView(rootEl, data, options) renders a /api/result payload
// (the same shape the owner page receives) as one .result-block per source
// image: SOURCE file-name line (owner pages), original/processed
// comparison, card rows with hover-zoom thumbnails, TCGplayer links,
// reference icons with lazy Scryfall previews, dual-face flip, set badges,
// candidate-set chips and the per-file total row at the bottom of the
// block.
//
// Above the blocks it renders a global view toolbar (one per page, at the
// top of the results list): a "Show thumbnails" switch (formerly the
// per-image "Show cards" toggle, now applied to every image at once), a
// "Show images" switch hiding the large original/detected comparison of
// every block, and a "Show set selection" switch (owner AND share pages
// alike) hiding the candidate-set chip rows (.set-candidates) and the set
// status badges (.set-badge) — handy once every ambiguous card has a picked
// set and the warnings and buttons get in the way. A picked card (the green
// "set selected" badge) hides outright; an unpicked warning badge ("set
// unclear" / "set unverified") collapses to a bare amber ⚠ (no pill, no
// background — see .set-selection-hidden in style.css) with its tooltip
// swapped to "no set was selected — the shown set may be wrong"
// (applySetSelectionTitles) so a card skipped by mistake still stands out
// in the clean list. All off leaves a bare list of card rows. On the owner
// page only (the page passes onChangeShowPrice) a short
// "Show price" switch sits beside them: it controls the price overlay on
// the exported (detected) images, so it needs a server-side re-render and
// the page owns its state (options.showPrice, re-passed on every render)
// instead of module state. The view choices live in module-level state so
// an automatic full re-render (append completion, failed-remove rollback,
// share snapshot) re-applies the user's current view mode instead of
// snapping back to the defaults.
//
// options:
//   readOnly (bool)   true on share pages: candidate-set chips become inert
//                     <span>s (same classes/attrs, hover preview kept), the
//                     .js-update-status element is omitted, the SOURCE:
//                     file-name header line is omitted entirely, and no
//                     /api/search-card or /api/update-processed requests are
//                     ever made. Cards marked set_picked (by the backend in
//                     the share snapshot) render the green "set selected"
//                     badge with the picked chip highlighted / siblings
//                     dimmed; hover-to-zoom works on both images on every
//                     page (attached here, not in app.js).
//   refBase  (string) URL prefix for in-stock Scryfall reference images:
//                     "/scryfall_images" on the owner page (default),
//                     "/share/<token>/ref" on share pages.
//   jobId    (string) owner page only; passed through to options.pick.
//   onBlock  (fn)     onBlock(block, item) after each block's DOM is built;
//                     the owner page uses it to attach its background
//                     re-render controller (makeImageUpdater).
//   pick     (fn)     pick(jobId, item, cardIndex, override) when the user
//                     picks a candidate set (owner page only; readOnly
//                     chips never fire it anyway).
//   onRemove (fn)     onRemove(jobId, filename) when the per-image remove
//                     button is clicked (owner page only: the button is
//                     rendered for non-readOnly views when this callback is
//                     provided; share pages pass neither, so remove controls
//                     never appear there).
//   onChangeFinish (fn) onChangeFinish(jobId, filename, cardIndex, finish)
//                     when a per-card finish toggle is clicked (owner page
//                     only; readOnly views render the toggle as inert
//                     <span>s and never fire it). finish is one of
//                     "nonfoil", "foil", "etched".
//   showPrice (bool)   owner page only: the checked state of the "Show
//                     price" switch (the page keeps the state itself and
//                     passes it on every render, e.g. from the /api/result
//                     payload's show_price flag). Ignored unless
//                     onChangeShowPrice is also provided.
//   onChangeShowPrice (fn) onChangeShowPrice(value) when the "Show price"
//                     switch is flipped (owner page only). The switch is
//                     rendered ONLY when this callback is provided: the
//                     read-only share page passes neither option, so the
//                     switch never appears there (a share snapshot can't be
//                     re-rendered server-side).
//   onTotalChange (fn) onTotalChange(filename, totalValue) after each block
//                     is built (its initial total) and again whenever a set
//                     pick or a finish toggle changes that block's total.
//                     The owner and share pages both sum these into their
//                     grand total (the #grandTotal span, moved into the view
//                     toolbar row via the totalEl option below; on the
//                     read-only share page the initial totals are final:
//                     nothing can change a price there).
//   totalEl  (element) the page's #grandTotal span (owner + share pages).
//                     Appended to the .view-toggles-row toolbar right after
//                     it is built, so the grand total shares the row with
//                     the two view toggles, pushed to the far right by CSS
//                     (.view-toggles-row .grand-total { margin-left: auto }).
//                     The toolbar is rebuilt on every render, so the element
//                     is re-appended each time; the page keeps its own
//                     reference to it for the value updates.
//
// Every .result-block carries dataset.filename = the source image's stored
// filename, so the page script can locate/remove a block without escaping
// attribute selectors.
//
// Plain script (no ES modules); exposes window.createResultsView.
// Load it BEFORE app.js.
(function () {
    "use strict";

    const TCGPLAYER_ICON = '<svg focusable="false" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14"><g fill="#000" fill-rule="evenodd"><path d="m2.715 3.004-1.981.285a.82.82 0 0 0-.714.926l.989 6.396c.07.445.5.752.964.687l.742-.106V3.004ZM13.287 3.289l-1.955-.28v8.187l.716.104c.462.066.894-.242.964-.687L14 4.218c.07-.446-.25-.862-.714-.927v-.002Z" fill-rule="nonzero"></path><path d="M4.094 2.1h5.86c.47 0 .848.366.848.816v8.368c0 .452-.38.816-.847.816H4.094c-.47 0-.847-.366-.847-.816V2.916c0-.452.38-.816.847-.816Zm3.56 4.063h1.78c.17 0 .271.186.176.321l-2.534 3.564c-.12.165-.388.086-.388-.115V8.084H4.612c-.157 0-.26-.158-.192-.294l1.75-3.48a.214.214 0 0 1 .192-.115h1.943c.157 0 .26.158.192.294l-.843 1.674Z"></path></g></svg>';

    // "Reference image" photo icon (source: resources/image-icon.svg),
    // shown to the left of a matched card's name; fill comes from CSS
    // (.card-ref-icon svg { fill: currentColor }, colored var(--primary)).
    const CARD_REF_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 122.88 97.68" aria-hidden="true"><path fill-rule="evenodd" clip-rule="evenodd" d="M0,0h122.88v97.68H0V0L0,0z M34.56,25.4c4.41,0,7.99,3.58,7.99,7.99c0,4.42-3.58,7.99-7.99,7.99 c-4.41,0-7.99-3.58-7.99-7.99C26.57,28.97,30.15,25.4,34.56,25.4L34.56,25.4z M68.2,59.7l15.99-27.64l16.99,42.96l-79.27,0v-5.33 l6.66-0.33l6.66-16.32l3.33,11.66h9.99l8.66-22.31L68.2,59.7L68.2,59.7z M9.13,8.09h104.63v81.49H9.13V8.09L9.13,8.09z"/></svg>';

    // "Flip arrows" icon (from resources/svg_icon.txt, the "double-faced {F}" entry),
    // shown as a small badge overlapping the reference icon for dual-face
    // cards — at-a-glance hint that the card has two faces. Same fill
    // treatment as the main icon (.card-ref-flip svg { fill: currentColor }).
    const CARD_FLIP_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" aria-hidden="true"><path d="M884.3,357.6c116.8,117.7,151.7,277-362.2,320V496.4L243.2,763.8L522,1031.3V860.8C828.8,839.4,1244.9,604.5,884.3,357.6z"></path><path d="M557.8,288.2v138.4l230.8-213.4L557.8,0v142.8c-309.2,15.6-792.1,253.6-426.5,503.8C13.6,527.9,30,330.1,557.8,288.2z"></path></svg>';

    // Global view-toggle state ("Show thumbnails" / "Show images"). Module
    // level so the choices survive the automatic full re-renders that
    // rebuild the whole results list; createResultsView re-applies them to
    // the fresh DOM on every render.
    let thumbnailsVisible = true;
    let imagesVisible = true;
    let setSelectionVisible = true;

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str == null ? "" : String(str);
        return div.innerHTML;
    }


    function createResultsView(rootEl, data, options) {
        options = options || {};
        const readOnly = options.readOnly === true;
        const refBase = options.refBase || "/scryfall_images";

        rootEl.innerHTML = "";

        // Global view toolbar: sits above every block, toggles the card
        // thumbnails and the large original/detected images for ALL images
        // at once (the old per-image "Show cards" switch is gone).
        const viewToolbar = document.createElement("div");
        viewToolbar.className = "view-toggles-row";
        // "Show price" (owner page only, see the options header): short
        // label so the switch fits the toolbar line next to the two view
        // toggles without wrapping or colliding with the grand total at the
        // far right. Checked state is page-owned (options.showPrice, the
        // server flag mirrored by the page) — not module state — because
        // the switch drives server-side re-renders, not just the view.
        const showPriceToggleHtml = typeof options.onChangeShowPrice === "function"
            ? `<label class="switch-label" title="Show or hide the price on the exported (detected) images">
                <input type="checkbox" class="switch-checkbox js-show-price-toggle"${options.showPrice !== false ? " checked" : ""}>
                <span class="switch"></span>
                Show price
            </label>`
            : "";
        viewToolbar.innerHTML = `
            <label class="switch-label" title="Show or hide the card thumbnails for every image">
                <input type="checkbox" class="switch-checkbox js-thumbnails-toggle"${thumbnailsVisible ? " checked" : ""}>
                <span class="switch"></span>
                Show thumbnails
            </label>
            <label class="switch-label" title="Show or hide the large original/detected images for every photo">
                <input type="checkbox" class="switch-checkbox js-images-toggle"${imagesVisible ? " checked" : ""}>
                <span class="switch"></span>
                Show images
            </label>
            <label class="switch-label" title="Show or hide the set selection buttons and the set status warnings (set unclear / set selected) — a card without a picked set keeps a bare ⚠">
                <input type="checkbox" class="switch-checkbox js-set-selection-toggle"${setSelectionVisible ? " checked" : ""}>
                <span class="switch"></span>
                Show set selection
            </label>
            ${showPriceToggleHtml}
        `;
        rootEl.appendChild(viewToolbar);
        // The grand total is parked in the page template (next to the
        // results container); move it into the toolbar so it sits on the
        // same row as the two view toggles, at the far right. Re-appended
        // on every re-render (the toolbar is rebuilt each time).
        if (options.totalEl) viewToolbar.appendChild(options.totalEl);
        const thumbsToggle = viewToolbar.querySelector(".js-thumbnails-toggle");
        thumbsToggle.addEventListener("change", (e) => {
            thumbnailsVisible = e.target.checked;
            rootEl.querySelectorAll(".js-card-list").forEach((list) =>
                list.classList.toggle("hide-thumbs", !thumbnailsVisible));
        });
        viewToolbar.querySelector(".js-images-toggle").addEventListener("change", (e) => {
            imagesVisible = e.target.checked;
            rootEl.classList.toggle("results-images-hidden", !imagesVisible);
        });
        // "Show set selection" OFF: the unpicked warning badges (set unclear
        // / set unverified) keep a bare ⚠ (the CSS collapses the pill to the
        // glyph), so swap their tooltip to the real problem — no set was
        // picked, the shown set may be wrong. Picked cards (the green
        // "set selected" badge, which carries no data-full-title) are hidden
        // outright and keep their own tooltip.
        function applySetSelectionTitles(hidden) {
            const NO_PICK_TITLE = "No set was selected for this card - the shown set may be wrong. Turn on 'Show set selection' to pick the right one.";
            rootEl.querySelectorAll(".set-badge").forEach((b) => {
                if (!b.dataset.fullTitle) return;
                b.title = hidden ? NO_PICK_TITLE : b.dataset.fullTitle;
            });
        }
        viewToolbar.querySelector(".js-set-selection-toggle").addEventListener("change", (e) => {
            setSelectionVisible = e.target.checked;
            rootEl.classList.toggle("set-selection-hidden", !setSelectionVisible);
            applySetSelectionTitles(!setSelectionVisible);
        });
        if (typeof options.onChangeShowPrice === "function") {
            viewToolbar.querySelector(".js-show-price-toggle").addEventListener("change", (e) => {
                options.onChangeShowPrice(e.target.checked);
            });
        }

        data.results.forEach((item) => {
            const block = document.createElement("div");
            block.className = "result-block";
            block.dataset.filename = item.filename;

            const totals = { value: 0 };
            // Displayed price per card, parallel to item.cards; used to
            // recompute the totals row when a candidate edition is picked.
            const shownPrices = [];
            const cardsHtml = item.cards
                .map((c, idx) => {
                    const thumb = c.crop_url
                        ? `<a class="card-thumb-wrap" href="${c.crop_url}" download title="Click to download this card image">
                            <img class="card-thumb" src="${c.crop_url}" alt="${escapeHtml(c.name)}">
                            <div class="card-thumb-full">
                                <img src="${c.crop_url}" alt="${escapeHtml(c.name)}">
                            </div>
                        </a>`
                        : "";
                    if (c.name === "Unmatched card") {
                        shownPrices[idx] = 0;
                        return `<div class="card-row">${thumb}<span class="card-name unmatched">Unmatched card</span></div>`;
                    }
                    const priceValue = c.price_usd ? parseFloat(c.price_usd) : 0;
                    shownPrices[idx] = priceValue;
                    totals.value += priceValue;
                    const price = priceValue ? `$${priceValue.toFixed(2)}` : "";
                    // The name is plain — no finish tag (the toggle beside it
                    // conveys the finish; the tag lives only on the image
                    // label and the CSV row).
                    const nameHtml = c.scryfall_uri
                        ? `<a class="card-name-link" href="${escapeHtml(c.scryfall_uri)}" target="_blank" rel="noopener noreferrer">${escapeHtml(c.name)}</a>`
                        : escapeHtml(c.name);
                    const tcgplayerLink = c.tcgplayer_purchase_uri
                        ? `<a class="tcgplayer-link" href="${escapeHtml(c.tcgplayer_purchase_uri)}" target="_blank" rel="noopener noreferrer" title="Buy on TCGplayer">${TCGPLAYER_ICON}</a>`
                        : "";
                    const metaParts = [];
                    if (c.card_type) metaParts.push(escapeHtml(c.card_type));
                    if (c.edition) metaParts.push(setChipIconHtml(c.set_code) + escapeHtml(c.set_code ? `${c.edition} [${c.set_code}]` : c.edition));
                    const cardMeta = metaParts.length ? `<span class="card-meta">${metaParts.join(" &middot; ")}</span>` : "";
                    // Phase 6: badge + candidate chips only for ambiguous/unverified
                    // multi-set cards; unique/resolved cards render as before.
                    const badgeHtml = setBadgeHtml(c);
                    const chipsHtml = setChipsHtml(c, readOnly, refBase);
                    return `<div class="card-row" data-card-index="${idx}">${thumb}<span class="card-name"><span class="card-name-line">${cardRefHtml(c, undefined, refBase)}${nameHtml}${finishToggleHtml(c, idx, readOnly)}${tcgplayerLink}${cardMeta}${badgeHtml}</span>${chipsHtml}</span><span class="price">${price}</span></div>`;
                })
                .join("");
            block.innerHTML = `
                ${readOnly ? "" : `
                <div class="file-name">
                    <div class="file-name-wrap">
                        <span>SOURCE: ${escapeHtml(item.filename)}</span>
                        <button type="button" class="remove-btn" title="Remove this image from the scan" aria-label="Remove ${escapeHtml(item.filename)} from the scan">&times;</button>
                    </div>
                </div>`}
                <div class="comparison-grid">
                    <div>
                        <div class="column-title">Original (hover to zoom)</div>
                        <div class="image-frame">
                            <img src="${item.original_url}" class="zoom-image js-zoom-img" alt="Original">
                        </div>
                    </div>
                    <div class="status-anchor">
                        <div class="column-title" style="color: var(--primary);">Detected &amp; Identified (hover to zoom)</div>
                        <div class="image-frame js-zoom-box">
                            <img src="${item.processed_url}" class="zoom-image js-zoom-img" alt="Processed">
                        </div>
                        ${readOnly ? "" : '<div class="update-status js-update-status" hidden></div>'}
                    </div>
                </div>
                <div class="card-list js-card-list">${cardsHtml || '<div class="card-row"><span class="card-name unmatched">No cards detected</span></div>'}</div>
                ${item.cards.length ? `<div class="total-row">Total value: $${totals.value.toFixed(2)} USD</div>` : ""}
            `;
            rootEl.appendChild(block);
            if (options.onBlock) options.onBlock(block, item);

            // Report this block's running total to the page, bound to this
            // image's filename (null when the page doesn't pass
            // onTotalChange). Fired for the initial total here, and again
            // from the set-pick / finish-toggle handlers below whenever this
            // block's total changes.
            const reportTotal = typeof options.onTotalChange === "function"
                ? (value) => options.onTotalChange(item.filename, value)
                : null;
            if (reportTotal) reportTotal(totals.value);

            // Per-image remove button (owner view only; the button itself is
            // only rendered for non-readOnly views, and only when a callback
            // is provided). The page script (app.js removeImage) does the
            // optimistic DOM removal + the /api/remove-image call.
            const removeBtn = block.querySelector(".remove-btn");
            if (removeBtn && options.onRemove) {
                removeBtn.addEventListener("click", () => options.onRemove(options.jobId, item.filename));
            }

            // Hover-to-zoom on BOTH images (original + processed), so it also
            // works on read-only share pages (moved here from app.js, which
            // the share page does not load).
            block.querySelectorAll(".comparison-grid .image-frame").forEach(attachZoomToFrame);

            const cardList = block.querySelector(".js-card-list");

            // Phase 6: candidate chips switch the displayed edition. Each
            // successful pick re-renders this image's on-image tags/prices
            // automatically in the background (see makeImageUpdater) - no
            // manual "Update" click is needed. The per-file total row at
            // the bottom of the block must track edition switches (and
            // finish toggles).
            const totalEls = block.querySelectorAll(".total-row");
                        // Owner mode only: forward set picks to the page-level pick
            // callback (options.pick), which on the owner page routes them
            // to the block's makeImageUpdater controller (created via
            // options.onBlock). Read-only views (share pages) get null:
            // no picks, no /api/update-processed requests.
            const imageUpdate = readOnly
                ? null
                : {
                    pick: (idx, override) => options.pick(options.jobId, item, idx, override),
                    finish: (idx, finishVal) =>
                        options.onChangeFinish && options.onChangeFinish(options.jobId, item.filename, idx, finishVal),
                };
            if (cardList) {
                cardList.querySelectorAll(".card-row[data-card-index]").forEach((row) => {
                    const idx = parseInt(row.dataset.cardIndex, 10);
                    const card = item.cards[idx];
                    row.querySelectorAll(".set-chip").forEach((chip) => {
                        // Share pages (readOnly): chips are inert spans and must
                        // never trigger /api/search-card (see header contract).
                        if (readOnly) return;
                        chip.addEventListener("click", () =>
                            switchCardEdition(row, chip, card, idx, totals, shownPrices, totalEls, imageUpdate, refBase, reportTotal));
                    });
                });
                // Hover previews (set chips + card-name reference icon):
                // load the in-stock Scryfall reference image on first hover;
                // if the file is missing, fall back to a "no reference image"
                // message inside the popup. Delegated on the card list so the
                // binding survives applyEditionToRow rewriting the name line.
                cardList.addEventListener("mouseover", (e) => {
                    const wrap = e.target.closest(".set-chip-wrap, .card-ref-wrap");
                    if (!wrap || !cardList.contains(wrap)) return;
                    const img = wrap.querySelector("img[data-src]");
                    if (!img || img.getAttribute("src")) return;
                    const empty = wrap.querySelector(".set-chip-preview-empty, .card-ref-preview-empty");
                    // Persistent (not once): a later flip to the other face
                    // (dual-face cards) must re-show the fallback if that
                    // file is missing too.
                    img.addEventListener("error", () => {
                        img.style.display = "none";
                        if (empty) empty.classList.add("show");
                    });
                    img.src = img.dataset.src;
                });
                // Dual-face cards: clicking the reference icon flips the
                // popup image between front and back
                // (scryfall_images/<id>.jpg <-> <id>_back.jpg). The flip
                // sticks on img.src, so the chosen face is still shown on
                // later hovers; clicking again flips back to the front.
                cardList.addEventListener("click", (e) => {
                    const icon = e.target.closest(".card-ref-icon");
                    if (!icon || !cardList.contains(icon)) return;
                    const wrap = icon.closest(".card-ref-wrap");
                    const img = wrap && wrap.querySelector("img.card-ref-preview-img");
                    const frontSrc = img && img.dataset.srcFront;
                    const backSrc = img && img.dataset.srcBack;
                    if (!img || !frontSrc || !backSrc) return;
                    const empty = wrap.querySelector(".card-ref-preview-empty");
                    const showBack = img.getAttribute("src") !== backSrc;
                    if (empty) empty.classList.remove("show");
                    img.style.display = "";
                    img.src = showBack ? backSrc : frontSrc;
                    icon.title = showBack
                        ? "Click to flip back to the front"
                        : "Click to see the other side";
                });
                // Per-card finish toggle (matched cards only): delegate on
                // the card list so the buttons keep working after
                // applyEditionToRow rewrites the name line. Clicking the
                // option the card already has is a no-op (no re-render).
                cardList.addEventListener("click", (e) => {
                    const btn = e.target.closest("button.finish-opt");
                    if (!btn || !cardList.contains(btn) || readOnly) return;
                    const row = btn.closest(".card-row[data-card-index]");
                    if (!row) return;
                    const i = parseInt(row.dataset.cardIndex, 10);
                    const card = item.cards[i];
                    if (!card || !card.scryfall_id) return; // unmatched: no toggle
                    const finishVal = btn.dataset.finish;
                    if (finishVal === (card.finish || "nonfoil")) return;
                    updateFinishInPlace(row, card, i, finishVal, totals, shownPrices, totalEls, imageUpdate, reportTotal);
                });
            }
        });

        // Re-apply the current global view mode to the freshly rendered
        // blocks (the DOM is rebuilt on every render, so the classes and
        // the container flag must be set again from the module state).
        rootEl.classList.toggle("results-images-hidden", !imagesVisible);
        rootEl.classList.toggle("set-selection-hidden", !setSelectionVisible);
        applySetSelectionTitles(!setSelectionVisible);
        if (!thumbnailsVisible) {
            rootEl.querySelectorAll(".js-card-list").forEach((list) =>
                list.classList.add("hide-thumbs"));
        }
    }

    // Hover-to-zoom for one image frame (original or processed): the
    // .zoomed class scales the inner img 2.2x (style.css) and the transform
    // origin follows the cursor, so the hovered area stays in place. Bound
    // by createResultsView on every frame, so it works on the owner page
    // and the read-only share pages alike (previously app.js-only).
    function attachZoomToFrame(frame) {
        const img = frame.querySelector("img");
        if (!img) return;
        frame.addEventListener("mousemove", (e) => {
            const rect = img.getBoundingClientRect();
            if (rect.width && rect.height) {
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;
                img.style.transformOrigin = `${x}% ${y}%`;
            }
            img.classList.add("zoomed");
        });
        frame.addEventListener("mouseleave", () => img.classList.remove("zoomed"));
    }

    // --- Set disambiguation (phase 6) ---
    // Multi-set artworks (same-artwork reprints) carry set_status / candidates /
    // symbol_margin in the /api/result JSON. 'unique' and 'resolved' cards render
    // exactly like before; 'ambiguous' and 'unverified' ones get a badge plus
    // candidate chips. Clicking a chip reloads that edition via the existing
    // POST /api/search-card endpoint and updates the row + totals row. Cards
    // from older jobs lack the new fields and render unchanged (every helper
    // below is gated on field presence).

    function finishSuffix(finish) {
        if (finish === "foil") return " (Foil)";
        if (finish === "etched") return " (Etched Foil)";
        return "";
    }

    // Per-card finish toggle: a small 3-option segmented control on the
    // name line of every matched card (normal / foil / etched), letting the
    // owner switch the card's finish after detection. Clicks are handled by
    // the delegated handler on .js-card-list, so the buttons keep working
    // after applyEditionToRow rewrites the name line. Read-only views
    // (share pages) render inert <span>s: the current finish stays visible
    // but nothing is clickable and onChangeFinish is never fired.
    const FINISH_TOGGLES = [
        { finish: "nonfoil", glyph: "N", label: "Normal" },
        { finish: "foil", glyph: "\u2726", label: "Foil" },
        { finish: "etched", glyph: "\u272A", label: "Etched Foil" },
    ];
    function finishToggleHtml(card, cardIndex, readOnly) {
        const active = (card && card.finish) || "nonfoil";
        const opts = FINISH_TOGGLES.map((t) => {
            const cls = "finish-opt" + (t.finish === active ? " active" : "");
            return readOnly
                ? `<span class="${cls}" data-finish="${t.finish}" title="${t.label}">${t.glyph}</span>`
                : `<button type="button" class="${cls}" data-card-index="${cardIndex}" data-finish="${t.finish}" title="${t.label}">${t.glyph}</button>`;
        });
        return `<span class="finish-toggle" title="Finish">${opts.join("")}</span>`;
    }

    function priceForFinish(card, finish) {
        // Mirror app.py's _finish_price (+ _pick_price_source) exactly, so
        // the optimistic re-pricing always matches what the server stores
        // (payload, image label, CSV):
        //  - the foil/etched price falls back to the nonfoil price when
        //    missing/zero (the server uses `or`, so 0 counts as missing);
        //  - the nonfoil base falls back to the card's current price_usd,
        //    EXCEPT for a manually picked card whose picked edition has no
        //    nonfoil price: the server prices picks from the picked edition
        //    only, so its base is null there;
        //  - legacy cards without the price triple use price_usd for every
        //    finish (the single detection/pick-time price).
        const hasTriple = card.price_usd_nonfoil !== undefined
            || card.price_usd_foil !== undefined
            || card.price_usd_etched !== undefined;
        let base = card.price_usd_nonfoil;
        if (base == null) {
            base = hasTriple && card.set_picked ? null : card.price_usd;
        }
        if (finish === "foil") return card.price_usd_foil || base || null;
        if (finish === "etched") return card.price_usd_etched || base || null;
        return base || null;
    }

    // The warning badges' text sits in a .set-badge-text span and their full
    // tooltip is stamped in data-full-title: when "Show set selection" is
    // OFF the CSS collapses an unpicked warning badge to a bare ⚠ (hiding
    // .set-badge-text) and applySetSelectionTitles swaps title to the
    // "no set was selected" warning (and back on).
    function setBadgeHtml(c) {
        if (!c || !c.candidates || c.candidates.length <= 1) return "";
        if (c.set_picked && (c.set_status === "ambiguous" || c.set_status === "unverified")) {
            // The owner picked this set: the backend marks the card in the
            // share snapshot (set_picked); the picked set's name comes from
            // the matching candidate (set_code fallback when missing).
            // Status-gated to mirror the owner page, where the green marker
            // only ever exists on cards that have a chip row.
            const chosen = c.candidates.find((k) => k.set_code === c.set_code);
            const setName = (chosen && chosen.set_name) || c.set_code || "?";
            return `<span class="set-badge set-badge-user" title="The scan owner picked this set for the card">set selected: ${escapeHtml(setName)}</span>`;
        }
        if (c.set_status === "ambiguous") {
            const margin = c.symbol_margin != null
                ? ` (margin ${parseFloat(c.symbol_margin).toFixed(2)})` : "";
            const title = "Two or more sets match the printed symbol too closely to call - pick one below";
            return `<span class="set-badge ambiguous" data-full-title="${escapeHtml(title)}" title="${escapeHtml(title)}"><span class="set-badge-text">&#9888; set unclear${margin}</span></span>`;
        }
        if (c.set_status === "unverified") {
            const title = "Set disambiguation is off or has no template for this card - pick the right set below";
            return `<span class="set-badge unverified" data-full-title="${escapeHtml(title)}" title="${escapeHtml(title)}"><span class="set-badge-text">set unverified</span></span>`;
        }
        return "";
    }

    function setChipIconHtml(setCode) {
        // Set-symbol icon (SET_ICONS from static/js/set_icons.js, same lookup
        // rules as search.html's setIconFor: case-insensitive). The printed
        // symbol is what distinguishes reprints, so it is shown to the left of
        // the set name for comparison against the actual card image.
        if (!setCode || typeof SET_ICONS === "undefined") return "";
        const svg = SET_ICONS[String(setCode).toUpperCase()];
        if (!svg) return "";
        return `<span class="set-chip-icon" aria-hidden="true">${svg}</span>`;
    }

    function cardRefHtml(c, forcedFace, refBase) {
        // Amber "reference image" icon to the left of the card name (matched
        // cards only). Hovering shows the in-stock Scryfall reference image
        // for this card (scryfall_images/<scryfall_id>.jpg, same naming as
        // /api/search-card's image_url) in the same popup style as the card
        // thumbnail hover, so it can be compared with the scanned crop. The
        // image is lazy-loaded on first hover (data-src). Initial-render
        // cards carry scryfall_id; /api/search-card replies (edition switch)
        // carry image_url instead, so derive the id from its file name.
        // Dual-face cards (dual_face flag) also carry the back image
        // (scryfall_images/<id>_back.jpg): clicking the icon flips the popup
        // to the other face (delegated click handler below).
        if (!c) return "";
        const refId = c.scryfall_id
            || (c.image_url ? c.image_url.split("/").pop().replace(/\.jpg$/i, "") : "");
        if (!refId) return "";
        const label = escapeHtml(c.name || "");
        const dual = !!c.dual_face;
        const frontUrl = `${refBase}/${encodeURIComponent(refId)}.jpg`;
        const backUrl = `${refBase}/${encodeURIComponent(refId)}_back.jpg`;
        // The popup opens on the face the photo matched (c.matched_face, so a
        // back-of-card upload compares side by side); after an edition switch
        // applyEditionToRow passes the face currently shown (forcedFace).
        let face = "front";
        if (dual) {
            if (forcedFace === "front" || forcedFace === "back") face = forcedFace;
            else if (c.matched_face === "back") face = "back";
        }
        const faceAttr = dual ? ` data-src-front="${frontUrl}" data-src-back="${backUrl}"` : "";
        const title = dual
            ? "Hover to compare with the reference image, click to see the other side"
            : "Hover to compare with the reference image";
        // Dual-face cards: small flip-arrows badge overlapping the icon's
        // bottom-right corner (it is nested inside .card-ref-icon, so clicks
        // on it hit the same delegated flip handler as the main icon).
        const flipBadge = dual
            ? `<span class="card-ref-flip" aria-hidden="true">${CARD_FLIP_ICON}</span>`
            : "";
        return `<span class="card-ref-wrap${dual ? " card-ref-dual" : ""}">` +
            `<span class="card-ref-icon" title="${title}">${CARD_REF_ICON}${flipBadge}</span>` +
            `<span class="card-thumb-full card-ref-preview">` +
            `<img class="card-ref-preview-img" data-src="${face === "back" ? backUrl : frontUrl}"${faceAttr} alt="${label}">` +
            `<span class="card-ref-preview-empty">No reference image in stock</span></span></span>`;
    }

    function setChipsHtml(c, readOnly, refBase) {
        if (!c || !c.candidates || c.candidates.length <= 1) return "";
        if (c.set_status !== "ambiguous" && c.set_status !== "unverified") return "";
        // Read-only views (share pages) render chips as inert <span>s with
        // the same class/data-attrs (hover preview kept); no click handler.
        const chipTag = readOnly ? "span" : "button";
        const picked = c.set_picked === true;
        const chips = c.candidates.map((k) => {
            const label = `${escapeHtml(k.set_name || k.set_code)} (${escapeHtml(k.set_code)})` +
                (k.score != null ? ` &middot; ${parseFloat(k.score).toFixed(2)}` : "");
            const active = k.set_code === c.set_code ? " active" : "";
            // Picked cards: green the picked chip, dim the rest. /api/result
            // marks set_picked for previously picked cards (owner + share),
            // so a full re-render (e.g. after an append) restores this
            // state; a live pick gets it in place via markSetChecked.
            const selCls = picked
                ? (k.set_code === c.set_code ? " set-chip-selected" : " set-chip-dimmed")
                : "";
            const scryfallId = k.scryfall_id || "";
            // Hover preview: the real Scryfall reference image we have in stock
            // for this card in this set (scryfall_images/<scryfall_id>.jpg, same
            // naming as /api/search-card's image_url), shown in the same popup
            // style as the card thumbnail hover so it can be compared with the
            // scanned crop. The image is lazy-loaded on first hover (data-src).
            const setLabel = escapeHtml(k.set_name || k.set_code) + " (" + escapeHtml(k.set_code) + ")";
            const previewImg = scryfallId
                ? `<img class="set-chip-preview-img" data-src="${refBase}/${encodeURIComponent(scryfallId)}.jpg" alt="${setLabel}">`
                : "";
            return `<span class="set-chip-wrap">` +
                `<${chipTag}${readOnly ? "" : ' type="button"'} class="set-chip${active}${selCls}" data-set-code="${escapeHtml(k.set_code)}" data-set-name="${escapeHtml(k.set_name || k.set_code)}" data-scryfall-id="${escapeHtml(scryfallId)}">${setChipIconHtml(k.set_code)}${label}</${chipTag}>` +
                `<span class="card-thumb-full set-chip-preview">${previewImg}` +
                `<span class="set-chip-preview-empty${previewImg ? "" : " show"}">No reference image in stock</span></span>` +
                `</span>`;
        });
        return `<span class="set-candidates">${chips.join("")}</span>`;
    }

    function applyEditionToRow(row, card, rc, refBase) {
        // Update one card row to the edition returned by /api/search-card.
        // Returns the new displayed price in dollars (0 when unpriced).
        const finish = card.finish || "nonfoil";
        const priceValue = priceForFinish(rc, finish);
        const nameLine = row.querySelector(".card-name-line");
        if (nameLine) {
            // Dual-face cards: carry the face the popup was showing over to
            // the re-rendered reference icon (/api/search-card replies carry
            // no matched_face, so the row's current face wins).
            const oldImg = nameLine.querySelector("img.card-ref-preview-img");
            let shownFace = null;
            if (oldImg && oldImg.dataset.srcBack) {
                const shown = oldImg.getAttribute("src") || oldImg.dataset.src;
                shownFace = shown === oldImg.dataset.srcBack ? "back" : "front";
            }
            // Plain name: the finish is conveyed by the toggle, not a name
            // suffix (same convention as the initial render).
            const nameHtml = rc.scryfall_uri
                ? `<a class="card-name-link" href="${escapeHtml(rc.scryfall_uri)}" target="_blank" rel="noopener noreferrer">${escapeHtml(rc.name)}</a>`
                : escapeHtml(rc.name);
            const tcgplayerLink = rc.tcgplayer_purchase_uri
                ? `<a class="tcgplayer-link" href="${escapeHtml(rc.tcgplayer_purchase_uri)}" target="_blank" rel="noopener noreferrer" title="Buy on TCGplayer">${TCGPLAYER_ICON}</a>`
                : "";
            const metaParts = [];
            if (rc.card_type) metaParts.push(escapeHtml(rc.card_type));
            if (rc.edition) metaParts.push(setChipIconHtml(rc.set_code) + escapeHtml(rc.set_code ? `${rc.edition} [${rc.set_code}]` : rc.edition));
            const cardMeta = metaParts.length ? `<span class="card-meta">${metaParts.join(" &middot; ")}</span>` : "";
            // The rebuild wipes the finish toggle, so re-add it (the
            // delegated click handler picks the fresh buttons up
            // automatically).
            const cardIndex = row.dataset.cardIndex !== undefined ? row.dataset.cardIndex : "";
            nameLine.innerHTML = `${cardRefHtml(rc, shownFace, refBase)}${nameHtml}${finishToggleHtml(card, cardIndex, false)}${tcgplayerLink}${cardMeta}${setBadgeHtml(card)}`;
        }
        row.querySelectorAll(".set-chip").forEach((chip) =>
            chip.classList.toggle("active", chip.dataset.setCode === rc.set_code));
        const priceEl = row.querySelector(".price");
        if (priceEl) priceEl.textContent = priceValue ? `$${parseFloat(priceValue).toFixed(2)}` : "";
        return priceValue ? parseFloat(priceValue) : 0;
    }

    function markSetChecked(row, chip) {
        // The user picked a set for this card: mark the row "checked" so long
        // result lists can be triaged at a glance - the picked chip goes green,
        // its siblings dim, and the amber/gray status badge is replaced in
        // place by a green "set selected" tag. The on-image label/price and
        // the CSV are re-rendered automatically in the background
        // (see makeImageUpdater).
        row.querySelectorAll(".set-chip").forEach((other) => {
            other.classList.toggle("set-chip-dimmed", other !== chip);
            other.classList.toggle("set-chip-selected", other === chip);
        });
        const nameLine = row.querySelector(".card-name-line");
        const badge = nameLine ? nameLine.querySelector(".set-badge") : null;
        if (badge) {
            badge.classList.remove("ambiguous", "unverified");
            badge.classList.add("set-badge-user");
            // Drop the warning badge's data-full-title: a picked card must
            // not get the "no set was selected" tooltip from
            // applySetSelectionTitles when "Show set selection" is toggled.
            badge.removeAttribute("data-full-title");
            badge.title = "You picked this set for the card";
            badge.textContent = `set selected: ${chip.dataset.setName || chip.dataset.setCode || ""}`;
        }
    }

    // Optimistically switch one card's finish in the results list: the
    // finish field, the price (re-priced locally from the card's price
    // triple) and the active toggle option are all updated in place (the
    // name stays plain — the toggle IS the finish indicator), then
    // imageUpdate.finish() queues the background re-render of the processed
    // image (labels + CSV, where the suffix reappears). The server stays
    // the source of truth: a failed re-render shows up on the image's
    // status line, and a full re-render from /api/result restores the
    // authoritative state.
    function updateFinishInPlace(row, card, idx, finishVal, totals, shownPrices, totalEls, imageUpdate, onTotalChange) {
        card.finish = finishVal;
        const line = row.querySelector(".card-name-line");
        if (line) {
            // Active state on the toggle options.
            line.querySelectorAll(".finish-opt").forEach((b) =>
                b.classList.toggle("active", b.dataset.finish === finishVal));
        }
        // Price + file totals (local re-pricing; the server re-prices the
        // same way in the re-rendered payload and CSV).
        const priceValue = priceForFinish(card, finishVal);
        card.price_usd = priceValue;
        const priceEl = row.querySelector(".price");
        if (priceEl) priceEl.textContent = priceValue ? `$${parseFloat(priceValue).toFixed(2)}` : "";
        const newShown = priceValue ? parseFloat(priceValue) : 0;
        if (totals) {
            totals.value += newShown - (shownPrices[idx] || 0);
            shownPrices[idx] = newShown;
            totalEls.forEach((el) => {
                el.textContent = `Total value: $${totals.value.toFixed(2)} USD`;
            });
            if (onTotalChange) onTotalChange(totals.value);
        }
        // Queue the background re-render (image labels + CSV).
        if (imageUpdate && typeof imageUpdate.finish === "function") {
            imageUpdate.finish(idx, finishVal);
        }
    }

    async function switchCardEdition(row, chip, card, idx, totals, shownPrices, totalEls, imageUpdate, refBase, onTotalChange) {
        if (row.dataset.switching === "1") return;
        row.dataset.switching = "1";
        chip.classList.add("loading");
        try {
            const fd = new FormData();
            fd.append("scryfall_id", chip.dataset.scryfallId || "");
            // Bare name (finish tag stripped) so the endpoint's fallback name
            // search still matches if the id lookup ever fails.
            const baseName = (card.name || "").replace(/\s*(\(Foil\)|\(Etched Foil\))\s*$/, "").trim();
            if (!baseName) throw new Error("no card name");
            fd.append("card_name", baseName);
            const resp = await fetch("/api/search-card", { method: "POST", body: fd });
            const data = await resp.json();
            if (!resp.ok || !data.success) {
                throw new Error(data.message || data.error || "Edition lookup failed");
            }
            const newPrice = applyEditionToRow(row, card, data.card, refBase);
            // The pick succeeded: mark the row "checked" (green) for triage.
            markSetChecked(row, chip);
            totals.value += newPrice - (shownPrices[idx] || 0);
            shownPrices[idx] = newPrice;
            totalEls.forEach((el) => {
                el.textContent = `Total value: $${totals.value.toFixed(2)} USD`;
            });
            if (onTotalChange) onTotalChange(totals.value);
            chip.title = "";
            // Fire the background re-render of the processed image: the
            // server persists the pick, so it also survives picks made while
            // this request is in flight (and picks made on other images).
            // Re-picking the set the row already showed BEFORE this click
            // (a confirmation click) leaves the on-image label untouched,
            // so it skips the re-render. Compared against the previously
            // shown set (row dataset, or the original job-result set),
            // then the dataset is advanced to the newly shown set.
            const shownSetCode = row.dataset.shownSetCode !== undefined
                ? row.dataset.shownSetCode
                : (card.set_code || "");
            const isConfirmation = shownSetCode !== "" && chip.dataset.setCode === shownSetCode;
            if (imageUpdate && !isConfirmation) {
                // picked_card carries the full picked-edition card, including
                // its price triple (the search endpoint's price_usd IS that
                // edition's nonfoil price, so it becomes price_usd_nonfoil):
                // a full re-render from /api/result (append completion,
                // failed-remove rollback, share page) restores the pick —
                // name/price/set from the override, edition/type/links/
                // reference image from picked_card — and later finish toggles
                // re-price from this triple (server-side via the persisted
                // override, client-side via the in-memory sync below).
                const picked = {
                    ...data.card,
                    price_usd_nonfoil: data.card.price_usd_nonfoil !== undefined
                        ? data.card.price_usd_nonfoil
                        : data.card.price_usd,
                };
                // Keep the in-memory card in sync with the picked edition
                // so a later finish toggle re-prices from the pick, not
                // from the original detection match (same rules as the
                // server's _result_payload: set_picked + picked triple).
                card.set_picked = true;
                card.price_usd_nonfoil = picked.price_usd_nonfoil;
                card.price_usd_foil = data.card.price_usd_foil;
                card.price_usd_etched = data.card.price_usd_etched;
                card.price_usd = newPrice;
                imageUpdate.pick(idx, {
                    label_name: data.card.name + finishSuffix(card.finish || "nonfoil"),
                    set_code: data.card.set_code || null,
                    price_usd: newPrice,
                    picked_card: picked,
                });
            }
            row.dataset.shownSetCode = data.card.set_code || "";
        } catch (err) {
            chip.title = `Could not load this edition: ${err.message || err}`;
        } finally {
            row.dataset.switching = "";
            chip.classList.remove("loading");
        }
    }

    window.createResultsView = createResultsView;
})();
