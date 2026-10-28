(function () {
    "use strict";

    // Inline SVG icons (TCGplayer / card ref / flip arrows) live in
    // results_view.js, together with the rendering code that uses them.

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const folderInput = document.getElementById("folderInput");
    const browseFilesBtn = document.getElementById("browseFilesBtn");
    const browseFolderBtn = document.getElementById("browseFolderBtn");
    const fileListEl = document.getElementById("fileList");
    const placeholder = document.getElementById("placeholderContent");
    const startBtn = document.getElementById("startBtn");
    const errorBox = document.getElementById("errorBox");

    const finishButtons = document.querySelectorAll("#finishToggle button");

    const uploadPanel = document.getElementById("uploadPanel");
    const progressPanel = document.getElementById("progressPanel");
    const resultsPanel = document.getElementById("resultsPanel");

    const progressFill = document.getElementById("progressFill");
    const progressStep = document.getElementById("progressStep");
    const progressCounter = document.getElementById("progressCounter");

    const resultsContainer = document.getElementById("resultsContainer");
    const csvDownloadLink = document.getElementById("csvDownloadLink");
    const downloadImageLink = document.getElementById("downloadImageLink");
    const scanAnotherBtn = document.getElementById("scanAnotherBtn");
    // "Add Images" button in the results header (null in the jsdom smoke
    // harnesses, which use a minimal DOM that predates the append UI).
    const addImagesBtn = document.getElementById("addImagesBtn");
    // Grand total over all detected cards (sum of the per-image totals),
    // shown in the results header next to the action buttons. Null in the
    // jsdom smoke harnesses' minimal DOM; null-guarded below.
    const grandTotalEl = document.getElementById("grandTotal");

    // Share UI (read-only snapshot of the current results). The share bar is
    // not in the static markup: it is created on the first successful share
    // and removed on reset, so nothing is visible before the user clicks
    // Share. Null-guarded for the jsdom smoke harnesses, which evaluate this
    // file against a minimal DOM that predates the share UI.
    const shareBtn = document.getElementById("shareBtn");
    let shareBarEl = null;
    let shareUrlEl = null;
    let shareCopyBtn = null;
    let shareExpiryEl = null;

    let selectedFiles = []; // array of { file: File, isFolder: boolean }
    let finish = "nonfoil";
    // "Show price" switch in the results toolbar (owner page only; the view
    // renders it only because this page passes onChangeShowPrice). Shows/
    // hides the price overlay on the exported (detected) images. The server
    // is the source of truth (per-item + job flags); this is the mirror
    // that initialises the switch on every render (from the /api/result
    // payload's show_price, default true) and is pushed to every image's
    // updater on change, which re-renders the images in the background.
    let showPriceVisible = true;
    let pollTimer = null;
    let currentJobId = null; // job whose results are on screen (for Share)
    let jobActive = false; // true while a finished job's results are on
                           // screen: the Scan button then appends to that job
                           // ("Add N Images to Scan") instead of a fresh scan
    // Per-image totals (filename -> $ value), kept in sync by
    // createResultsView's onTotalChange callback (initial block totals plus
    // set picks and finish toggles); the results-header grand total is their
    // sum. Rebuilt on every renderResults, pruned on optimistic removal.
    const blockTotals = new Map();

    const IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];

    function isImage(name) {
        const lower = name.toLowerCase();
        return IMAGE_EXT.some((ext) => lower.endsWith(ext));
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.style.display = msg ? "block" : "none";
    }

    function updateFileList() {
        fileListEl.innerHTML = "";
        if (selectedFiles.length === 0) {
            placeholder.style.display = "flex";
            fileListEl.style.display = "none";
            fileListEl.classList.remove("thumbnail-grid");
            startBtn.disabled = true;
            return;
        }
        placeholder.style.display = "none";
        fileListEl.style.display = "block";

        // Folder uploads can contain a lot of files, so just list their names;
        // otherwise show an image thumbnail for each picked file.
        const hasFolder = selectedFiles.some((s) => s.isFolder);
        fileListEl.classList.toggle("thumbnail-grid", !hasFolder);
        selectedFiles.forEach((s, index) => {
            if (hasFolder) {
                const div = document.createElement("div");
                div.textContent = s.file.webkitRelativePath || s.file.name;
                fileListEl.appendChild(div);
            } else {
                const item = document.createElement("div");
                item.className = "thumbnail-item";
                const removeBtn = document.createElement("button");
                removeBtn.type = "button";
                removeBtn.className = "thumbnail-remove";
                removeBtn.textContent = "\u00d7";
                removeBtn.title = "Remove";
                removeBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    removeFile(index);
                });
                const img = document.createElement("img");
                img.src = URL.createObjectURL(s.file);
                img.alt = s.file.name;
                const label = document.createElement("span");
                label.textContent = s.file.name;
                item.appendChild(removeBtn);
                item.appendChild(img);
                item.appendChild(label);
                fileListEl.appendChild(item);
            }
        });
        startBtn.disabled = false;
        const n = selectedFiles.length;
        startBtn.textContent = jobActive
            ? `Add ${n} Image${n > 1 ? "s" : ""} to Scan`
            : `Scan ${n} Image${n > 1 ? "s" : ""}`;
    }

    function addFiles(fileArray, isFolder) {
        const valid = fileArray.filter((f) => isImage(f.name));
        if (valid.length === 0) return;
        selectedFiles = selectedFiles.concat(valid.map((file) => ({ file, isFolder: !!isFolder })));
        updateFileList();
        showError("");
    }

    function removeFile(index) {
        selectedFiles.splice(index, 1);
        updateFileList();
    }

    // --- Browse buttons ---
    browseFilesBtn.addEventListener("click", () => fileInput.click());
    browseFolderBtn.addEventListener("click", () => folderInput.click());
    dropZone.addEventListener("click", (e) => {
        if (e.target === dropZone || e.target === placeholder || placeholder.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", () => {
        addFiles(Array.from(fileInput.files), false);
        fileInput.value = "";
    });
    folderInput.addEventListener("change", () => {
        addFiles(Array.from(folderInput.files), true);
        folderInput.value = "";
    });

    // --- Drag & drop (files and folders) ---
    ["dragenter", "dragover"].forEach((evt) =>
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        })
    );
    ["dragleave", "drop"].forEach((evt) =>
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        })
    );

    function traverseEntry(entry) {
        return new Promise((resolve) => {
            if (entry.isFile) {
                entry.file(
                    (file) => resolve([file]),
                    () => resolve([])
                );
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                const allEntries = [];
                const readBatch = () => {
                    reader.readEntries(async (entries) => {
                        if (!entries.length) {
                            const nested = await Promise.all(allEntries.map(traverseEntry));
                            resolve(nested.flat());
                            return;
                        }
                        allEntries.push(...entries);
                        readBatch();
                    }, () => resolve([]));
                };
                readBatch();
            } else {
                resolve([]);
            }
        });
    }

    dropZone.addEventListener("drop", async (e) => {
        const items = e.dataTransfer.items;
        if (items && items.length && items[0].webkitGetAsEntry) {
            const entries = Array.from(items)
                .map((item) => item.webkitGetAsEntry())
                .filter(Boolean);
            const plainFiles = [];
            const folderFiles = [];
            for (const entry of entries) {
                const files = await traverseEntry(entry);
                if (entry.isDirectory) {
                    folderFiles.push(...files);
                } else {
                    plainFiles.push(...files);
                }
            }
            addFiles(plainFiles, false);
            addFiles(folderFiles, true);
        } else {
            addFiles(Array.from(e.dataTransfer.files), false);
        }
    });

    // --- Finish toggle ---
    finishButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            finishButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            finish = btn.dataset.finish;
        });
    });

    // --- Submit: fresh scan (POST /api/process) or append to the active job
    // (POST /api/process/<job_id>/add). The active job is the one whose
    // results are on screen (jobActive). An append failure keeps the previous
    // results on screen — never reset to the upload panel, which would hide
    // them. ---
    function buildScanFormData() {
        const formData = new FormData();
        selectedFiles.forEach((s) => formData.append("images", s.file, s.file.webkitRelativePath || s.file.name));
        formData.append("finish", finish);
        // No show_price here: the price-overlay choice moved to the results
        // toolbar (see showPriceVisible / onChangeShowPrice below), and the
        // server defaults it to ON for every new batch.
        return formData;
    }

    // The server now owns the batch: clear the selection so the drop zone
    // (and the "Add N Image(s) to Scan" counter) only ever counts NEW images
    // — an append must not re-upload the already-scanned ones. On a FAILED
    // upload the selection is kept, so the user can retry without re-picking.
    function clearFileSelection() {
        selectedFiles = [];
        updateFileList();
    }

    function switchToProgress() {
        uploadPanel.style.display = "none";
        resultsPanel.style.display = "none";
        progressPanel.style.display = "block";
        progressFill.style.width = "2%";
        progressStep.textContent = "Uploading images...";
        progressCounter.textContent = "";
        dropZone.classList.remove("append-hint");
    }

    function showResultsPanel() {
        progressPanel.style.display = "none";
        resultsPanel.style.display = "block";
    }

    async function runScan() {
        jobActive = false;
        if (addImagesBtn) addImagesBtn.hidden = true;
        showError("");
        switchToProgress();
        try {
            const resp = await fetch("/api/process", { method: "POST", body: buildScanFormData() });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.error || "Upload failed.");
            }
            clearFileSelection();
            // The address bar now carries the session URL (/job/<id>):
            // copyable, and re-attaches to THIS session after a tab loss or
            // an app restart (window.RESUMED_JOB_ID on page load, see the
            // resumeSession hook at the bottom). The job already exists on
            // the server — the URL identifies it, it does not create it.
            history.pushState({}, "", "/job/" + data.job_id);
            pollStatus(data.job_id);
        } catch (err) {
            resetToUpload();
            showError(err.message || String(err));
        }
    }

    async function runAppend() {
        if (!currentJobId) return;
        showError("");
        switchToProgress();
        try {
            const resp = await fetch(`/api/process/${currentJobId}/add`, { method: "POST", body: buildScanFormData() });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.error || "Could not add images to the scan.");
            }
            clearFileSelection();
            pollStatus(currentJobId, true);
        } catch (err) {
            // The previous results are still on the server: go back to them
            // and keep append mode — do NOT reset to the upload panel.
            showResultsPanel();
            showError(err.message || String(err));
        }
    }

    startBtn.addEventListener("click", () => {
        if (selectedFiles.length === 0) return;
        if (jobActive && currentJobId) runAppend();
        else runScan();
    });

    function pollStatus(jobId, isAppend = false) {
        pollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`/api/status/${jobId}`);
                if (resp.status === 404) {
                    // The job vanished (server restart / 48 h cleanup): exit
                    // append mode, there is no session left to return to.
                    jobActive = false;
                    if (addImagesBtn) addImagesBtn.hidden = true;
                    updateFileList(); // relabel the (re-enabled) Scan button
                    clearInterval(pollTimer);
                    resetToUpload();
                    showError("The scan session is no longer available. Please scan again.");
                    return;
                }
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || "Status check failed.");

                const total = data.total || 1;
                // Job-wide percentage from the backend (advances per detected
                // card); fall back to the old image-index ratio if absent.
                const pct = data.progress != null
                    ? Math.min(99, Math.max(0, data.progress))
                    : Math.min(98, Math.round((data.current / total) * 100));
                progressFill.style.width = `${Math.max(pct, 5)}%`;
                progressStep.textContent = data.step || "Working...";
                progressCounter.textContent = data.total ? `${data.current} / ${data.total} image(s)` : "";

                if (data.status === "done") {
                    clearInterval(pollTimer);
                    progressFill.style.width = "100%";
                    loadResults(jobId);
                } else if (data.status === "error") {
                    clearInterval(pollTimer);
                    // An append failure keeps the previous results on screen;
                    // a fresh-scan failure resets to the upload screen.
                    if (isAppend) showResultsPanel();
                    else resetToUpload();
                    showError(data.error || "Processing failed.");
                }
            } catch (err) {
                clearInterval(pollTimer);
                if (isAppend) showResultsPanel();
                else resetToUpload();
                showError(err.message || String(err));
            }
        }, 600);
    }

    async function loadResults(jobId) {
        const resp = await fetch(`/api/result/${jobId}`);
        const data = await resp.json();
        if (!resp.ok) {
            resetToUpload();
            showError(data.error || "Could not load results.");
            return;
        }
        renderResults(data, jobId);
        showResultsPanel();
    }

    // --- Session resume: `GET /job/<job_id>` re-renders this page with the
    // job id injected as window.RESUMED_JOB_ID (templates/index.html), so
    // after a tab loss or a server restart the owner can re-attach to the
    // existing session from its URL: live progress if it is still
    // queued/running, results if done (plus the existing share bar when one
    // exists), a friendly message if it is gone (48 h expiry / cleanup).
    async function resumeSession(jobId) {
        switchToProgress();
        progressStep.textContent = "Loading session...";
        try {
            let resp;
            try {
                resp = await fetch(`/api/status/${jobId}`);
            } catch (err) {
                // Network failure: the server is unreachable — nothing to
                // re-attach to from here.
                resetToUpload();
                showError("This scan is no longer available (it may have expired after 48 hours).");
                return;
            }
            if (resp.status === 404) {
                // The job vanished (48 h cleanup / unknown id — the HTML page
                // itself would already be the 404 session_gone page, this
                // covers a stale tab hitting the API directly).
                resetToUpload();
                showError("This scan is no longer available (it may have expired after 48 hours).");
                return;
            }
            const data = await resp.json();
            if (!resp.ok) {
                resetToUpload();
                showError(data.error || "This scan is no longer available.");
                return;
            }
            if (data.status === "queued" || data.status === "running") {
                // Still processing on the server: continue the live progress
                // (pollStatus's own 404/error/done handling takes over).
                pollStatus(jobId);
            } else if (data.status === "done") {
                await loadResults(jobId);
                reShowShareBarIfAny(jobId);
            } else if (data.status === "error") {
                resetToUpload();
                showError(data.error || "Processing failed.");
            }
        } catch (err) {
            resetToUpload();
            showError(err.message || String(err));
        }
    }

    // "Download Image" header link always targets the FIRST processed image;
    // refresh it whenever the results change (render or removal).
    function refreshDownloadLink(results) {
        const firstDownloadUrl = results.find((r) => r.processed_download_url)?.processed_download_url;
        if (firstDownloadUrl) {
            downloadImageLink.href = firstDownloadUrl;
            downloadImageLink.style.display = "inline-block";
        } else {
            downloadImageLink.style.display = "none";
        }
    }

    // --- Grand total (sum of all per-image totals) ---
    // The grand-total span is moved into the .view-toggles-row toolbar by
    // createResultsView (see results_view.js); anything that wipes
    // #resultsContainer (New Scan) detaches it. Park it back next to the
    // container, where the next render will move it into the toolbar again.
    function parkGrandTotal() {
        if (grandTotalEl && !grandTotalEl.isConnected && resultsContainer.parentNode) {
            resultsContainer.parentNode.insertBefore(grandTotalEl, resultsContainer);
        }
    }

    function updateGrandTotal() {
        if (!grandTotalEl) return;
        let sum = 0;
        blockTotals.forEach((value) => { sum += value; });
        grandTotalEl.textContent = `Total value: $${sum.toFixed(2)} USD`;
        grandTotalEl.hidden = false;
    }

    function resetGrandTotal() {
        blockTotals.clear();
        if (grandTotalEl) grandTotalEl.hidden = true;
    }

    function renderResults(data, jobId) {
        currentJobId = jobId;
        jobActive = true;
        if (addImagesBtn) addImagesBtn.hidden = false;
        resultsContainer.innerHTML = "";
        // Rebuilt by the onTotalChange callbacks as the blocks render below
        // (clearing first so removed images don't linger in the sum).
        resetGrandTotal();
        // "Show price" switch: mirror the server's job-level price-overlay
        // flag (the payload's show_price; the `!== false` keeps pre-flag
        // payloads and missing values on the ON default).
        showPriceVisible = data.show_price !== false;

        if (data.results.length) {
            // All result rendering is delegated to the shared results view
            // (results_view.js, loaded before this file). One background
            // re-render controller (makeImageUpdater) is created per
            // source-image block via onBlock; the pick callback routes the
            // view's set picks (switchCardEdition, in results_view.js) to the
            // matching controller. onRemove routes the per-image remove
            // button (owner view only) to removeImage.
            const updaters = new Map(); // source-image item -> image updater
            createResultsView(resultsContainer, data, {
                readOnly: false,
                refBase: "/scryfall_images",
                jobId,
                totalEl: grandTotalEl,
                onBlock: (block, item) => {
                    updaters.set(item, makeImageUpdater(jobId, item, block));
                },
                pick: (jId, item, cardIndex, override) => {
                    const update = updaters.get(item);
                    if (update) update.pick(cardIndex, override);
                },
                onChangeFinish: (jId, filename, cardIndex, finishVal) => {
                    const it = data.results.find((r) => r.filename === filename);
                    const update = it && updaters.get(it);
                    if (update) update.finish(cardIndex, finishVal);
                },
                // "Show price" toolbar switch (owner page only, see the
                // options header in results_view.js): re-renders every
                // processed image in the background — one request per block,
                // coalesced per image by its updater, each block dimming its
                // frame + showing its status while in flight.
                showPrice: showPriceVisible,
                onChangeShowPrice: (value) => {
                    showPriceVisible = value;
                    for (const update of updaters.values()) update.showPrice(value);
                },
                onTotalChange: (filename, value) => {
                    blockTotals.set(filename, value);
                    updateGrandTotal();
                },
                onRemove: removeImage,
            });
        } else {
            // Every source image was removed: keep the results panel alive
            // (header + Add Images button) so the session can continue.
            const note = document.createElement("div");
            note.className = "results-empty";
            note.textContent = "No images in this scan \u2014 use the Add Images button above to continue.";
            resultsContainer.appendChild(note);
        }

        if (data.csv_url) {
            csvDownloadLink.href = data.csv_url;
            csvDownloadLink.style.display = "inline-block";
        } else {
            csvDownloadLink.style.display = "none";
        }

        refreshDownloadLink(data.results);
        // (hover-to-zoom is attached by results_view.js on both image frames,
        // so it also works on the read-only share page)
    }

    // Per-image removal (owner view only; share pages never load app.js).
    // Optimistic: the block disappears immediately; on failure the full
    // results are re-fetched and re-rendered, which rolls the removal back
    // (server state is the source of truth).
    async function removeImage(jobId, filename) {
        if (!jobId || !filename) return;
        for (const block of resultsContainer.querySelectorAll(".result-block")) {
            if (block.dataset.filename === filename) {
                block.remove();
                break;
            }
        }
        // Keep the header grand total in sync with the optimistic removal;
        // on failure loadResults() re-renders and rebuilds the totals.
        blockTotals.delete(filename);
        updateGrandTotal();
        try {
            const resp = await fetch(`/api/remove-image/${jobId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || "Could not remove the image.");
            if (!data.results.length) {
                // Last image gone: swap the blocks for the empty state (the
                // header + Add Images button stay, the session survives).
                renderResults(data, jobId);
            } else {
                refreshDownloadLink(data.results);
            }
        } catch (err) {
            showError(err.message || String(err));
            loadResults(jobId);
        }
    }

    // --- Add Images: show the upload panel above the current results so the
    // user can pick more files; the (relabelled) Scan button then appends to
    // the current job instead of starting a fresh scan. ---
    if (addImagesBtn) addImagesBtn.addEventListener("click", () => {
        showError("");
        resultsPanel.style.display = "block";
        uploadPanel.style.display = "block";
        dropZone.classList.add("append-hint");
        if (typeof uploadPanel.scrollIntoView === "function") {
            uploadPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });


    // One controller per source image for the automatic processed-image
    // re-render. Every successful set pick calls pick() and every per-card
    // finish toggle calls finish(); both store their choice in per-image
    // pending maps and POST /api/update-processed/<job_id> in the
    // background. The page is never blocked: while a request is in flight,
    // further picks/finishes just mark the image "dirty" and one follow-up
    // re-render fires when it returns. Requests always carry the full
    // current pick + finish sets, and the server persists both per job, so
    // the displayed image always converges to "every pick and finish made
    // so far", no matter how quick the clicks are.
    function makeImageUpdater(jobId, item, block) {
        const pending = new Map(); // card index -> {label_name, set_code, price_usd}
        const pendingFinishes = new Map(); // card index -> "nonfoil"|"foil"|"etched"
        // Price-overlay choice (results toolbar "Show price"); undefined =
        // never flipped since this render, so the request omits show_price
        // and the server keeps the item's own flag. Like picks/finishes it
        // is cumulative (never cleared on success): a flip made while a
        // request is in flight would otherwise be wiped by that request's
        // success handler before the dirty-follow-up re-render fires.
        // Re-sending the current value in later requests is idempotent.
        let pendingShowPrice = undefined;
        const state = { inflight: false, dirty: false };
        const statusEl = block.querySelector(".js-update-status");
        const frame = block.querySelector(".js-zoom-box");

        function setStatus(text, cls) {
            if (!statusEl) return;
            statusEl.textContent = text || "";
            statusEl.className = "update-status js-update-status" + (cls ? " " + cls : "");
            statusEl.hidden = !text;
        }

        function reloadImage() {
            // Bust the cache so the browser actually reloads the re-rendered
            // image (the server overwrites the same file).
            const v = `v=${Date.now()}`;
            const img = frame ? frame.querySelector("img") : null;
            if (img) img.src = item.processed_url + (item.processed_url.includes("?") ? "&" : "?") + v;
            // The results-header "Download Image" link targets the first
            // processed image; refresh its cache buster if this is that one.
            const dlHref = downloadImageLink.getAttribute("href") || "";
            if (item.processed_download_url && dlHref.split("?")[0] === item.processed_download_url) {
                downloadImageLink.href = `${item.processed_download_url}?${v}`;
            }
        }

        async function runUpdate() {
            // Nothing new to send (defensive: pick/finish/showPrice always
            // fill their slots before calling, so this only guards stray
            // replays).
            if (pending.size === 0 && pendingFinishes.size === 0 && pendingShowPrice === undefined) return;
            state.inflight = true;
            if (frame) frame.classList.add("updating");
            setStatus("Updating detected image\u2026", "updating");
            try {
                const resp = await fetch(`/api/update-processed/${jobId}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: item.filename,
                        cards: Array.from(pending.entries()).map(([i, o]) => ({
                            card_index: i,
                            label_name: o.label_name,
                            set_code: o.set_code,
                            price_usd: o.price_usd,
                            // New-style picks carry the full picked-edition
                            // card (incl. its price triple) so the server
                            // can restore it on full re-renders and
                            // re-price finish toggles from the pick.
                            ...(o.picked_card ? { picked_card: o.picked_card } : {}),
                        })),
                        finishes: Array.from(pendingFinishes.entries()).map(([i, f]) => ({
                            card_index: i,
                            finish: f,
                        })),
                        // Set only when the "Show price" switch flipped
                        // (the server persists it on the item + job flags);
                        // omitted otherwise so the item's own flag stands.
                        ...(pendingShowPrice === undefined ? {} : { show_price: pendingShowPrice }),
                    }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || "Image update failed.");
                reloadImage();
                setStatus("", "");
                // pendingShowPrice stays (cumulative, like picks/finishes):
                // a flip made in flight is still pending for the
                // dirty-follow-up below, and re-sending the current value in
                // later requests is idempotent.
            } catch (err) {
                // The pick stays in `pending` and a pending show-price flip
                // stays in pendingShowPrice: the next re-render (triggered
                // by a later pick/flip on this image) will carry them again.
                // Keep the error visible until then.
                setStatus(`Could not update the detected image: ${err.message || err}`, "error");
            } finally {
                if (frame) frame.classList.remove("updating");
                state.inflight = false;
                if (state.dirty) {
                    state.dirty = false;
                    runUpdate();
                }
            }
        }

        return {
            // Store one manual pick and re-render the image in the background.
            pick(idx, override) {
                pending.set(idx, override);
                if (state.inflight) state.dirty = true;
                else runUpdate();
            },
            // Store one per-card finish choice and re-render the image in
            // the background (same coalescing as picks; the request always
            // carries the full current finish set for this image).
            finish(idx, finishVal) {
                pendingFinishes.set(idx, finishVal);
                if (state.inflight) state.dirty = true;
                else runUpdate();
            },
            // Flip this image's price overlay (results toolbar "Show price"
            // switch, called for every block from the page). Same coalescing
            // as picks/finishes: a flip while a request is in flight just
            // marks the image dirty and one follow-up re-render carries the
            // latest value, so quick on-off-on toggling converges.
            showPrice(value) {
                pendingShowPrice = value;
                if (state.inflight) state.dirty = true;
                else runUpdate();
            },
        };
    }

    // --- Share link ---
    // The server returns only the path; the absolute URL is composed from
    // location.origin so the same page works on the test tunnel and on the
    // real domain. Re-clicking Share refreshes the snapshot under the same
    // token (server-side idempotent) and updates the expiry hint.
    let shareRequestInFlight = false;
    let shareCopyResetTimer = null;

    function setCopyLabel(text) {
        if (shareCopyBtn) shareCopyBtn.textContent = text;
    }

    // Builds the share bar skeleton (URL + Copy + expiry hint) and inserts it
    // right before the results container. Only called after a successful
    // share or a resume re-show, so the bar never exists in the DOM before.
    function createShareBar() {
        if (shareBarEl) return shareBarEl;
        const bar = document.createElement("div");
        bar.id = "shareBar";
        bar.className = "share-bar";
        bar.innerHTML =
            '<span id="shareUrl" class="share-url"></span>' +
            '<button type="button" id="shareCopyBtn" class="btn-outline btn-small">Copy</button>' +
            '<span id="shareExpiry" class="share-expiry"></span>';
        shareUrlEl = bar.querySelector("#shareUrl");
        shareCopyBtn = bar.querySelector("#shareCopyBtn");
        shareExpiryEl = bar.querySelector("#shareExpiry");
        shareCopyBtn.addEventListener("click", copyShareUrl);
        if (resultsContainer && resultsContainer.parentNode) {
            resultsContainer.parentNode.insertBefore(bar, resultsContainer);
        } else {
            document.body.appendChild(bar);
        }
        shareBarEl = bar;
        return bar;
    }

    async function copyShareUrl() {
        if (!shareUrlEl) return;
        const url = shareUrlEl.textContent;
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(url);
            } else {
                const input = document.createElement("input");
                input.value = url;
                document.body.appendChild(input);
                input.select();
                document.execCommand("copy");
                input.remove();
            }
            setCopyLabel("Copied \u2713");
            if (shareCopyResetTimer) clearTimeout(shareCopyResetTimer);
            shareCopyResetTimer = setTimeout(() => setCopyLabel("Copy"), 2000);
        } catch (err) {
            setCopyLabel("Copy"); // URL stays visible; select it manually
        }
    }

    async function requestShare() {
        if (!currentJobId || shareRequestInFlight) return;
        shareRequestInFlight = true;
        if (shareBtn) {
            shareBtn.disabled = true;
            shareBtn.textContent = "Sharing\u2026";
        }
        try {
            const resp = await fetch(`/api/share/${currentJobId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: "{}",
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || "Could not create the share link.");
            applyShareInfo(data);
        } catch (err) {
            showError(err.message || String(err));
        } finally {
            shareRequestInFlight = false;
            if (shareBtn) {
                shareBtn.disabled = false;
                shareBtn.textContent = "Share";
            }
        }
    }

    // Fills (or updates) the share bar from a {share_path, expires_hint}
    // object — the reply shape of BOTH the Share click (POST /api/share/<id>)
    // and the session-page share info (GET /api/share/<id>), so the bar can
    // be re-shown on a resumed session without re-clicking Share.
    function applyShareInfo(data) {
        createShareBar();
        shareUrlEl.textContent = location.origin + data.share_path;
        shareExpiryEl.textContent =
            "available until ~" + new Date(data.expires_hint).toLocaleString();
    }

    // Re-shows an EXISTING share bar when a resumed session re-attaches to a
    // done job: a pure read (GET /api/share/<job_id> — no re-snapshot, no
    // token rotation, the 48 h clock is untouched). Only called on the
    // resume path: a fresh scan shows the bar only after the Share click.
    // 404 (no share) or any failure → no bar, silently.
    async function reShowShareBarIfAny(jobId) {
        if (!jobId) return;
        try {
            const resp = await fetch(`/api/share/${jobId}`);
            if (resp.status === 404 || !resp.ok) return;
            const data = await resp.json();
            if (data && data.share_path) applyShareInfo(data);
        } catch (err) {
            // No share (or the fetch failed): nothing to re-show.
        }
    }

    function hideShareBar() {
        currentJobId = null;
        if (shareBarEl) {
            shareBarEl.remove();
            shareBarEl = null;
        }
        shareUrlEl = null;
        shareCopyBtn = null;
        shareExpiryEl = null;
        if (shareCopyResetTimer) {
            clearTimeout(shareCopyResetTimer);
            shareCopyResetTimer = null;
        }
    }

    if (shareBtn) shareBtn.addEventListener("click", requestShare);

    function resetToUpload() {
        progressPanel.style.display = "none";
        resultsPanel.style.display = "none";
        uploadPanel.style.display = "block";
        resetGrandTotal();
        // A fresh scan starts with the server's price-overlay default (ON);
        // the first render re-mirrors the new job's flag from the payload
        // anyway.
        showPriceVisible = true;
    }

    // --- New-scan confirmation modal (themed in-page dialog) ---
    const newScanModal = document.getElementById("newScanModal");
    const newScanCancelBtn = document.getElementById("newScanCancelBtn");
    const newScanConfirmBtn = document.getElementById("newScanConfirmBtn");

    function startNewScan() {
        // Back to the home URL ("/"): the session URL (/job/<id>) no longer
        // describes this page. The old session is untouched on the server and
        // stays resumable at its URL — the wipe here is client-side only.
        history.pushState({}, "", "/");
        selectedFiles = [];
        updateFileList();
        resultsContainer.innerHTML = "";
        // The grand total now lives inside the container (view-toolbar row),
        // so the wipe detaches it — park it back next to the container; the
        // next render moves it into the toolbar again.
        parkGrandTotal();
        hideShareBar();
        jobActive = false; // leave append mode: next scan starts fresh
        if (addImagesBtn) addImagesBtn.hidden = true;
        dropZone.classList.remove("append-hint");
        resetToUpload();
    }

    function closeNewScanModal() {
        if (newScanModal) newScanModal.hidden = true;
    }

    if (newScanModal) {
        // Clicking the dimmed backdrop (outside the dialog box) cancels.
        newScanModal.addEventListener("click", (e) => {
            if (e.target === newScanModal) closeNewScanModal();
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && !newScanModal.hidden) closeNewScanModal();
        });
        if (newScanCancelBtn) {
            newScanCancelBtn.addEventListener("click", closeNewScanModal);
        }
        if (newScanConfirmBtn) {
            newScanConfirmBtn.addEventListener("click", () => {
                closeNewScanModal();
                startNewScan();
            });
        }
    }

    scanAnotherBtn.addEventListener("click", () => {
        // The page is wiped on a new scan (results, share bar, selected
        // files) — confirm in the themed in-page modal first.
        if (newScanModal) {
            newScanModal.hidden = false;
            // Focus Cancel (not the destructive action) so Enter doesn't wipe.
            if (newScanCancelBtn) newScanCancelBtn.focus();
            return;
        }
        // No modal markup (minimal jsdom harness DOMs): keep the legacy
        // confirm() guard — jsdom's no-op returns undefined, so the reset
        // proceeds there as before.
        if (typeof confirm === "function" &&
            !confirm("Are you sure? Starting a new scan will wipe the current results.")) {
            return;
        }
        startNewScan();
    });

    updateFileList();

    // Session resume hook: pages served by `GET /job/<job_id>` inject
    // window.RESUMED_JOB_ID (templates/index.html); a plain "/" load has it
    // null and the jsdom harnesses' minimal DOMs leave it undefined — a
    // truthiness check, so neither resumes.
    if (window.RESUMED_JOB_ID) resumeSession(window.RESUMED_JOB_ID);
})();
