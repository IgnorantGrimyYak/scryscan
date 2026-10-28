// share.js — read-only viewer for shared scans (GET /share/<token> pages).
//
// templates/share.html sets window.SHARE = { token, sharedAt } inline before
// this script loads. We fetch the stored snapshot payload from
// GET /api/share/<token>/result and render it with the shared results
// renderer (results_view.js) in readOnly mode: candidate-set chips are
// inert spans and no /api/search-card or /api/update-processed requests
// are ever made. Reference images load through the token-scoped
// /share/<token>/ref/... endpoint, so the link works without any access
// to the owner's app state.
//
// The results header shows the grand total over all detected cards (sum of
// the per-image totals), wired up through createResultsView's onTotalChange
// exactly like the owner page. In readOnly mode no edition picks or finish
// toggles can fire, so the total is final once the snapshot has rendered.
//
// On 404 (unknown/expired token) or any network failure the results panel
// is hidden and the "expired" message is shown instead.
(function () {
    "use strict";

    const EXPIRED_MSG = "This link has expired or is no longer available.";

    const share = window.SHARE || {};
    const token = share.token || "";
    const panel = document.getElementById("resultsPanel");
    const container = document.getElementById("resultsContainer");
    const errorBox = document.getElementById("errorBox");
    const grandTotalEl = document.getElementById("grandTotal");

    // Per-image running totals, keyed by filename; the grand total (moved
    // into the view-toolbar row by createResultsView) is their sum (same
    // accumulation the owner page does in app.js).
    const blockTotals = new Map();
    function updateGrandTotal() {
        if (!grandTotalEl) return;
        let sum = 0;
        blockTotals.forEach((v) => { sum += v; });
        grandTotalEl.textContent = `Total value: $${sum.toFixed(2)} USD`;
        grandTotalEl.hidden = false;
    }

    // The shared stylesheet keeps #resultsPanel hidden until the owner page's
    // app.js reveals it; on this page nothing else would, so reveal it here
    // (the "Loading shared scan…" placeholder becomes visible at once).
    if (panel) panel.style.display = "block";

    function showError(message) {
        if (panel) panel.style.display = "none";
        if (errorBox) {
            errorBox.style.display = "block";
            errorBox.textContent = message;
        }
    }

    function showExpired() {
        showError(EXPIRED_MSG);
    }

    (async () => {
        if (!token) {
            showExpired();
            return;
        }
        let data;
        try {
            const resp = await fetch(`/api/share/${encodeURIComponent(token)}/result`);
            if (resp.status === 404 || !resp.ok) {
                showExpired();
                return;
            }
            data = await resp.json();
        } catch (err) {
            // Network failure / non-JSON body: treat as unavailable.
            showExpired();
            return;
        }
        try {
            window.createResultsView(container, data, {
                readOnly: true,
                refBase: `/share/${encodeURIComponent(token)}/ref`,
                totalEl: grandTotalEl,
                onTotalChange: (filename, value) => {
                    blockTotals.set(filename, value);
                    updateGrandTotal();
                },
            });
        } catch (err) {
            // A renderer failure must never leave a silent blank page.
            console.error("share page render failed:", err);
            showError("Could not display the shared results. This link may be expired" +
                      " or the scan may have been cleaned up.");
        }
    })();
})();
