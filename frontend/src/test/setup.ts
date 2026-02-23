import "@testing-library/jest-dom/vitest";

// jsdom does not implement scrollIntoView — stub it globally
if (typeof window !== "undefined") {
  window.HTMLElement.prototype.scrollIntoView = function () {};
}
