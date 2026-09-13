function sendValue(value) {
  Streamlit.setComponentValue(value);
}

async function parseClipboardData() {
  try {
    const items = await navigator.clipboard.read();
    let imageBlob = null;
    for (const item of items) {
      for (const type of item.types) {
        if (type.startsWith("image/")) {
          imageBlob = await item.getType(type);
          break;
        }
      }
      if (imageBlob) break;
    }

    if (imageBlob) {
      const reader = new FileReader();
      reader.readAsDataURL(imageBlob);
      reader.onloadend = function () {
        sendValue(reader.result);
      };
    } else {
      console.error("No image found in clipboard.");
      sendValue("error: no image found in clipboard");
    }
  } catch (error) {
    console.error("Clipboard read error:", error);
    sendValue("error: " + error);
  }
}

function updateHeight() {
  const btn = document.getElementById("clover_paste_btn");
  if (btn) {
    const h = Math.max(btn.offsetHeight, 42);
    Streamlit.setFrameHeight(h + 2);
  }
}

function onRender(event) {
  const args = event.detail.args || {};
  const label = args.label || "Paste Screenshot";
  const textColor = args.text_color || "#FFFFFF";
  const bgColor = args.background_color || "#111827";
  const hoverBg = args.hover_background_color || "#1F2937";

  const isMac = /Mac|iPod|iPhone|iPad/i.test(navigator.userAgent || navigator.platform || "");
  const shortcut = isMac ? "Cmd+V" : "Ctrl+V";

  let displayLabel = label.replace(/Cmd\+V|Ctrl\+V|⌘V/gi, shortcut);

  const labelEl = document.getElementById("btn_label_text");
  if (labelEl) {
    labelEl.textContent = displayLabel;
  }

  const btn = document.getElementById("clover_paste_btn");
  if (btn) {
    btn.style.color = textColor;
    btn.style.backgroundColor = bgColor;

    if (!btn._listenersAttached) {
      btn.addEventListener("click", parseClipboardData);
      btn.addEventListener("mouseover", function() {
        btn.style.backgroundColor = hoverBg;
      });
      btn.addEventListener("mouseout", function() {
        btn.style.backgroundColor = bgColor;
      });
      btn._listenersAttached = true;
    }
  }

  updateHeight();
}

window.addEventListener("resize", updateHeight);
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(44);
