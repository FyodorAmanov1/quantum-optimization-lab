// Optional UI interaction smoke check using only Node's built-in modules.
// Run: node tests/studio_ui_smoke.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
class Element {
  constructor() { this.children = []; this.dataset = {}; }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  setAttribute() {}
}
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement() { return new Element(); },
  querySelectorAll() { return []; },
  addEventListener() {},
};
const context = vm.createContext({ document, clearInterval() {}, charts: [] });
// Exercise the real render/budget handlers, without network loading or SVG DOM.
const code = fs.readFileSync(path.join(root, "docs/app.js"), "utf8")
  .replace(/\nload\(\);\s*$/, "");
vm.runInContext(code, context);
vm.runInContext("renderChart = (chart) => charts.push(chart)", context);
const study = JSON.parse(fs.readFileSync(path.join(root, "docs/study.json"), "utf8"));
context.testCase = study.cases.find((c) => c.id === "hubbard-shots");
vm.runInContext("render(testCase)", context);
assert.equal(context.charts.length, 5);
assert.match(context.charts[0].xlabel, /20,480/);
const label = elements.get("charts").children[0];
const selector = label.children[0];
assert.equal(selector.value, "4096");
assert.equal(selector.children.length, 3);
for (const [value, total] of [["256", "1,280"], ["65536", "327,680"]]) {
  selector.value = value;
  context.charts.length = 0;
  selector.onchange();
  assert.equal(context.charts.length, 5);
  assert.match(context.charts[4].title, /binomial/);
  assert.match(context.charts[0].xlabel, new RegExp(total));
  assert.equal(elements.get("charts").children[0], label);
}
context.testCase = study.cases[0];
context.charts.length = 0;
vm.runInContext("render(testCase)", context);
assert.equal(context.charts.length, study.cases[0].charts.length);
assert.equal(elements.get("charts").children.length, 0);
console.log("Studio budget switching and legacy case rendering passed.");
