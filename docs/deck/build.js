// Builds the project review deck: node build.js <repo> <out.pptx>  (see build.sh)
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(process.env.PPTX_SKILL + '/scripts/apply_theme.js');

const REPO = process.argv[2];
const OUT = process.argv[3];
const BUILD = path.join(__dirname, 'build'); // generated data and figures (git-ignored)
const D = JSON.parse(fs.readFileSync(path.join(BUILD, 'deck_data.json')));
const PR = JSON.parse(fs.readFileSync(path.join(BUILD, 'pr_curves.json')));
const fig = (f) => path.join(REPO, 'report_figs', f);
const img = (f) => path.join(BUILD, 'img', f);

const THEME = {
  name: 'Night Flight',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1F2A3D', lt1: 'FFFFFF', dk2: '0E1A2F', lt2: 'EEF2F7',
    accent1: '2A78D6', accent2: 'EB6834', accent3: '1BAF7A', accent4: 'EDA100',
    accent5: 'E87BA4', accent6: '4A3AA7', hlink: '2A78D6', folHlink: '4A3AA7',
  },
};
const T = THEME.colors;
// hex-only places (chart colours, grid lines, backgrounds) read from THEME
const HEX = { ink: T.dk1, night: T.dk2, tint: T.lt2, muted: '52514E', faint: 'E3E2DE', sky: 'C9D6EA', gray: '9C9A92' };
// one colour per model on every chart (validated: dataviz validate_palette.js, light mode)
const MODEL = {
  coco_yolo11n: { label: 'COCO yolo11n (no training)', short: 'COCO yolo11n', color: HEX.gray },
  yolo11n_1frame: { label: 'yolo11n, 1 frame', short: 'n, 1 frame', color: T.accent4 },
  yolo11n_3frame: { label: 'yolo11n, 3 frames', short: 'n, 3 frames', color: T.accent3 },
  'yolo11n-p2_3frame': { label: 'yolo11n-p2, 3 frames', short: 'n-p2, 3 frames', color: T.accent5 },
  yolo11s_3frame: { label: 'yolo11s, 3 frames, 12 epochs', short: 's, 12 epochs', color: T.accent1 },
  yolo11s_3frame_p13_e50: { label: 'yolo11s, 3 frames, 50 epochs', short: 's, 50 epochs', color: T.accent2 },
};

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5
pres.author = 'Rohan Shenoy';
pres.title = 'Detecting aircraft in flight';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;
const M = 0.6; // side margin
const CW = W - 2 * M; // content width
const TOP = 1.8; // content top on light slides
const BOTTOM = 6.85; // content bottom (footer below)

// ---------- layouts ----------
const footer = [
  { text: { text: 'Detecting aircraft in flight · PlanesNet detector review', options: {
    x: M, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: HEX.muted, margin: 0 } } },
];
pres.defineSlideMaster({
  title: 'TITLE', background: { color: HEX.night },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 1.7, w: 6.6, h: 2.2, fontSize: 44, bold: true,
      color: C.background1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: M, y: 4.1, w: 6.4, h: 1.9, fontSize: 18,
      color: C.background2, valign: 'top', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'SECTION', background: { color: HEX.night },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 2.2, y: 2.75, w: 10.3, h: 1.0, fontSize: 40, bold: true,
      color: C.background1, valign: 'middle', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 2.2, y: 3.8, w: 10.3, h: 1.2, fontSize: 18,
      color: C.background2, valign: 'top', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: 'FFFFFF' },
  margin: [0.5, M, 0.7, M],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: CW, h: 0.75, fontSize: 36, bold: true,
      color: C.text1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: M, y: 1.12, w: CW, h: 0.55, fontSize: 18,
      color: C.accent1, valign: 'top', margin: 0 }, text: '' } },
    ...footer,
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: HEX.muted, align: 'right' },
});
pres.defineSlideMaster({
  title: 'CLOSING', background: { color: HEX.night },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: CW, h: 0.75, fontSize: 36, bold: true,
      color: C.background1, valign: 'bottom', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: M, y: 1.12, w: CW, h: 0.55, fontSize: 18,
      color: C.background2, valign: 'top', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: HEX.sky, align: 'right' },
});

// ---------- helpers ----------
let section = '';
function content(title, kicker, notes) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section });
  s.addText(title, { placeholder: 'title' });
  s.addText(kicker, { placeholder: 'body', bullet: false });
  if (notes) s.addNotes(notes);
  return s;
}
let sectionNo = 0;
function divider(title, desc, notes) {
  sectionNo += 1;
  section = title;
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: 'SECTION', sectionTitle: section });
  s.addShape(pres.shapes.OVAL, { x: M + 0.1, y: 2.85, w: 1.1, h: 1.1, fill: { color: C.accent2 }, line: { type: 'none' },
    objectName: 'Section badge' });
  s.addText(String(sectionNo), { x: M + 0.1, y: 2.85, w: 1.1, h: 1.1, align: 'center', valign: 'middle', fontSize: 34,
    bold: true, color: C.background1, fontFace: THEME.headFontFace, isTextBox: true, margin: 0, objectName: 'Section number' });
  s.addText(title, { placeholder: 'title' });
  s.addText(desc, { placeholder: 'body', bullet: false });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, x, y, w, h, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 },
    line: { type: 'none' }, objectName: opts.name || 'Card' });
}
function stat(s, x, y, w, h, big, label, opts = {}) {
  card(s, x, y, w, h, { fill: opts.fill, name: 'Stat card' });
  s.addText(big, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.95, fontSize: opts.bigSize || 48, bold: true,
    color: opts.bigColor || C.accent2, fontFace: THEME.headFontFace, isTextBox: true, margin: 0, valign: 'bottom',
    objectName: 'Stat value' });
  s.addText(label, { x: x + 0.25, y: y + 1.22, w: w - 0.5, h: h - 1.4, fontSize: opts.labelSize || 14,
    color: opts.labelColor || C.text1, isTextBox: true, margin: 0, valign: 'top', objectName: 'Stat label' });
}
function bullets(s, items, x, y, w, h, opts = {}) {
  const runs = items.map((t, i) => {
    const o = { bullet: opts.numbered ? { type: 'number' } : true, breakLine: i < items.length - 1,
      paraSpaceAfter: opts.space || 8 };
    if (Array.isArray(t)) return t.map((r, j) => ({ text: r.text, options: Object.assign({}, j === 0 ? o : {},
      r.options || {}, j === t.length - 1 ? { breakLine: o.breakLine } : {}) }));
    return [{ text: t, options: o }];
  }).flat();
  s.addText(runs, { x, y, w, h, fontSize: opts.size || 16, color: opts.color || C.text1, valign: 'top', isTextBox: true,
    margin: 0, objectName: opts.name || 'Bullets' });
}
function caption(s, text, x, y, w, h = 0.35, opts = {}) {
  s.addText(text, { x, y, w, h, fontSize: opts.size || 12, color: opts.color || C.text1, italic: !!opts.italic,
    align: opts.align || 'left', valign: 'top', isTextBox: true, margin: 0, objectName: opts.name || 'Caption' });
}
function picture(s, file, x, y, w, aspect, alt) {
  s.addImage({ path: file, x, y, w, h: w / aspect, altText: alt, objectName: alt });
  return y + w / aspect;
}
function chartOpts(extra) {
  return Object.assign({
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt',
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLineShow: false, valAxisLineShow: false,
    valGridLine: { color: HEX.faint, size: 0.75 }, catGridLine: { style: 'none' },
    dataLabelColor: HEX.ink, dataLabelFontFace: '+mn-lt', dataLabelFontSize: 12,
    legendFontFace: '+mn-lt', legendFontSize: 12, legendColor: HEX.ink, showLegend: false,
    titleFontFace: '+mn-lt', titleFontSize: 14, titleColor: HEX.ink, showTitle: true,
    valAxisTitleColor: HEX.muted, catAxisTitleColor: HEX.muted, valAxisTitleFontSize: 12, catAxisTitleFontSize: 12,
    valAxisTitleFontFace: '+mn-lt', catAxisTitleFontFace: '+mn-lt',
  }, extra);
}
function table(s, header, rows, x, y, w, colW, opts = {}) {
  const fs = opts.size || 13;
  const head = header.map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text2 },
    fontSize: fs, valign: 'middle' } }));
  const body = rows.map((r, i) => r.map((c) => (typeof c === 'object' ? c : { text: String(c), options: {
    fontSize: fs, color: C.text1, valign: 'top', fill: { color: i % 2 ? C.background1 : C.background2 } } })));
  s.addTable([head, ...body], { x, y, w, colW, border: { type: 'none' }, margin: [0.06, 0.1, 0.06, 0.1],
    rowH: opts.rowH, autoPage: false, objectName: opts.name || 'Table' });
}
const pct = (v) => Math.round(v * 100) + '%';
const p2 = D.part2;

module.exports = { pres, C, T, HEX, MODEL, D, PR, fig, img, M, CW, TOP, BOTTOM, W, content, divider, card, stat, bullets,
  caption, picture, chartOpts, table, pct, p2, THEME, applyTheme, OUT, setSection: (t) => { section = t; } };
require('./slides.js');
