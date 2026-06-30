document.querySelectorAll("figure img").forEach((image) => {
  image.addEventListener("error", () => {
    const placeholder = document.createElement("div");
    placeholder.className = "figure-missing";
    placeholder.innerHTML = "Figure not generated yet.<br><code>python -m scripts.make_figures</code>";
    image.replaceWith(placeholder);
  });
});

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", () => {
    history.replaceState(null, "", link.getAttribute("href"));
  });
});
