document.addEventListener(
  "click",
  function (event) {
    var button = event.target.closest(".md-clipboard");
    if (!button) return;
    var block = button.closest(".language-pycon");
    if (!block) return;
    var code = block.querySelector("pre code") || block.querySelector("code");
    if (!code) return;
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
