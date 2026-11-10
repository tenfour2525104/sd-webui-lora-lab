// Lora Lab: thumbnail card selection.
// Clicking a card toggles it in the selection, immediately paints the
// selection frame + sequence number in the browser, and mirrors the
// selection (as JSON) into the hidden bridge textbox of the extension.
// The Python side listens to that field and reflects the selection into
// the Lora slots above.
window.__loraLabState = window.__loraLabState || {};

function loraLabEnsureInit(container) {
    var bridgeId = container.getAttribute("data-bridge");
    var token = container.getAttribute("data-token");
    var st = window.__loraLabState[bridgeId];
    if (!st || st.token !== token) {
        var sel = [];
        try {
            sel = JSON.parse(container.getAttribute("data-selected") || "[]");
            if (!Array.isArray(sel)) sel = [];
        } catch (e) {
            sel = [];
        }
        window.__loraLabState[bridgeId] = { token: token, sel: sel };
    }
    return window.__loraLabState[bridgeId];
}

function loraLabRepaint(container, names, sel) {
    var cards = container.querySelectorAll(".lora-lab-card");
    cards.forEach(function (card) {
        var idx = parseInt(card.getAttribute("data-idx"), 10);
        var name = names[idx];
        var pos = sel.indexOf(name);
        var badge = card.querySelector(".lora-lab-badge");
        if (pos >= 0) {
            card.classList.add("lora-lab-selected");
            if (badge) {
                badge.textContent = String(pos + 1);
                badge.style.display = "flex";
            }
        } else {
            card.classList.remove("lora-lab-selected");
            if (badge) {
                badge.textContent = "";
                badge.style.display = "none";
            }
        }
    });
}

function loraLabToggle(card) {
    var container = card.closest(".lora-lab-grid");
    if (!container) return;
    var st = loraLabEnsureInit(container);
    var names = [];
    try {
        names = JSON.parse(container.getAttribute("data-names") || "[]");
    } catch (e) {
        return;
    }
    var idx = parseInt(card.getAttribute("data-idx"), 10);
    var name = names[idx];
    if (name === undefined) return;
    var pos = st.sel.indexOf(name);
    if (pos >= 0) {
        st.sel.splice(pos, 1);
    } else {
        st.sel.push(name);
    }
    loraLabRepaint(container, names, st.sel);
    var bridgeId = container.getAttribute("data-bridge");
    var wrap = document.getElementById(bridgeId);
    if (!wrap) return;
    var ta = wrap.querySelector("textarea") || wrap.querySelector("input");
    if (!ta) return;
    ta.value = JSON.stringify(st.sel);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    ta.dispatchEvent(new Event("change", { bubbles: true }));
}
