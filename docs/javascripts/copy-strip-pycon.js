(function () {
  function bind(block) {
    var button = block.querySelector(".md-clipboard");
    if (!button || button.dataset.pyconStripBound) return;
    button.dataset.pyconStripBound = "1";
    button.addEventListener(
      "click",
      function (event) {
        var code = block.querySelector("pre code");
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
  }

  function setup() {
    document.querySelectorAll(".language-pycon").forEach(bind);
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(setup);
  } else {
    document.addEventListener("DOMContentLoaded", setup);
  }
})();
