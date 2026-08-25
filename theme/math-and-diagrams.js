/* Client-side KaTeX math and Mermaid diagrams for mdBook.
 *
 * mdBook has no native $-delimiter math. This script renders:
 *   $$ ... $$   display math (may span multiple lines within one block)
 *   $ ... $     inline math, ONLY when the content has no leading/trailing
 *               whitespace (the Pandoc rule). That rule is what keeps prose
 *               like "costs $2 and $15 per million" from being eaten: the
 *               candidate span "2 and " ends in a space, so it is rejected.
 *
 * Mermaid: fenced ```mermaid blocks arrive as <pre><code class="language-mermaid">.
 * They are swapped to <pre class="mermaid"> and rendered, with the diagram
 * theme following the book's light/dark theme.
 */
(function () {
  "use strict";

  var KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/";
  var MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js";

  function loadCss(href) {
    var l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = href;
    document.head.appendChild(l);
  }

  function loadScript(src, onload) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    document.head.appendChild(s);
  }

  /* ---------- math ---------- */

  var SKIP = { CODE: 1, PRE: 1, SCRIPT: 1, STYLE: 1, TEXTAREA: 1, SVG: 1 };

  function textNodesUnder(root) {
    var nodes = [];
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        for (var p = n.parentNode; p; p = p.parentNode) {
          if (p.nodeType === 1) {
            if (SKIP[p.nodeName.toUpperCase()]) return NodeFilter.FILTER_REJECT;
            if (p.classList && p.classList.contains("katex")) return NodeFilter.FILTER_REJECT;
          }
        }
        return n.nodeValue.indexOf("$") !== -1
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    });
    var n;
    while ((n = walker.nextNode())) nodes.push(n);
    return nodes;
  }

  // Split a text node's content into literal text and math segments.
  function segments(text) {
    var out = [];
    var i = 0;
    var lit = "";

    function flush() {
      if (lit) out.push({ math: null, text: lit });
      lit = "";
    }

    while (i < text.length) {
      if (text[i] === "\\" && text[i + 1] === "$") {
        lit += "$";                       // escaped dollar: \$ -> literal $
        i += 2;
        continue;
      }
      if (text[i] === "$" && text[i + 1] === "$") {
        var close = text.indexOf("$$", i + 2);
        if (close !== -1) {
          flush();
          out.push({ math: text.slice(i + 2, close), display: true });
          i = close + 2;
          continue;
        }
      }
      if (text[i] === "$") {
        var j = i + 1;
        while (j < text.length && text[j] !== "$") {
          if (text[j] === "\\") j++;      // skip escaped chars inside math
          j++;
        }
        if (j < text.length) {
          var body = text.slice(i + 1, j);
          // The Pandoc rule: real inline math hugs its delimiters.
          var ok =
            body.length > 0 &&
            !/^\s/.test(body) &&
            !/\s$/.test(body) &&
            !/^\d/.test(text[j + 1] || ""); // "$5$4" style money-adjacent noise
          if (ok) {
            flush();
            out.push({ math: body, display: false });
            i = j + 1;
            continue;
          }
        }
      }
      lit += text[i];
      i++;
    }
    flush();
    return out;
  }

  function renderMath() {
    var nodes = textNodesUnder(document.body);
    nodes.forEach(function (node) {
      var segs = segments(node.nodeValue);
      if (!segs.some(function (s) { return s.math != null; })) return;

      var frag = document.createDocumentFragment();
      segs.forEach(function (s) {
        if (s.math == null) {
          frag.appendChild(document.createTextNode(s.text));
        } else {
          var span = document.createElement(s.display ? "div" : "span");
          try {
            katex.render(s.math, span, {
              displayMode: !!s.display,
              throwOnError: false,
            });
          } catch (e) {
            span.textContent = (s.display ? "$$" : "$") + s.math + (s.display ? "$$" : "$");
          }
          frag.appendChild(span);
        }
      });
      node.parentNode.replaceChild(frag, node);
    });
  }

  /* ---------- mermaid ---------- */

  function bookIsDark() {
    var cls = document.documentElement.className || "";
    return /\b(navy|coal|ayu)\b/.test(cls);
  }

  function renderMermaid() {
    var blocks = document.querySelectorAll("pre > code.language-mermaid");
    if (!blocks.length) return;
    blocks.forEach(function (code) {
      var pre = code.parentNode;
      var div = document.createElement("pre");
      div.className = "mermaid";
      div.textContent = code.textContent;
      pre.parentNode.replaceChild(div, pre);
    });
    loadScript(MERMAID, function () {
      mermaid.initialize({
        startOnLoad: false,
        theme: bookIsDark() ? "dark" : "default",
        securityLevel: "strict",
      });
      mermaid.run({ querySelector: ".mermaid" });
    });
  }

  /* ---------- boot ---------- */

  loadCss(KATEX + "katex.min.css");
  loadScript(KATEX + "katex.min.js", function () {
    renderMath();
  });
  renderMermaid();
})();
