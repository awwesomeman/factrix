document.addEventListener(
  "click",
  function (event) {
    var button = event.target.closest('[data-md-type="copy"]');
    if (!button) return;
    var targetSel = button.getAttribute("data-clipboard-target");
    if (!targetSel) return;
    var code = document.querySelector(targetSel);
    if (!code) return;
    if (!code.closest(".language-pycon")) return;
    var clone = code.cloneNode(true);
    clone.querySelectorAll(".gp, .go").forEach(function (node) {
      node.remove();
    });
    var text = clone.textContent
      .replace(/^[ \t]*\n/gm, "")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/\s+$/, "\n");
    event.stopImmediatePropagation();
    event.preventDefault();
    navigator.clipboard.writeText(text);
  },
  true,
);
