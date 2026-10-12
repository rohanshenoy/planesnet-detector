const B = require('./build.js');
const { pres, C, T, HEX, MODEL, D, PR, fig, img, M, CW, TOP, BOTTOM, W, content, divider, card, stat, bullets, caption,
  picture, chartOpts, table, pct, p2 } = B;

// ================= title =================
B.setSection('Overview');
pres.addSection({ title: 'Overview' });
{
  const s = pres.addSlide({ masterName: 'TITLE', sectionTitle: 'Overview' });
  s.addText('Detecting aircraft in flight', { placeholder: 'title' });
  s.addText([
    { text: 'From a parked-plane classifier to satellite and sky detectors, measured on real aircraft in flight',
      options: { breakLine: true } },
    { text: ' ', options: { breakLine: true, fontSize: 10 } },
    { text: 'Rohan Shenoy · October 2026', options: { fontSize: 14, color: HEX.sky } },
  ], { placeholder: 'body', bullet: false });
  s.addImage({ path: img('s2_flying_examples.png'), x: 7.55, y: 2.15, w: 5.2, h: 5.2 / 2.727,
    altText: 'Aircraft in flight on Sentinel-2 at ten airports', objectName: 'Title image' });
  s.addText('Aircraft in flight on Sentinel-2, 10 m pixels: each shows as blue, green and red copies',
    { x: 7.55, y: 2.25 + 5.2 / 2.727, w: 5.2, h: 0.5, fontSize: 12, color: HEX.sky, isTextBox: true, margin: 0,
      objectName: 'Title image caption' });
  s.addNotes('Review of the planesnet-detector work: rebuilding a PlanesNet classifier (parked aircraft, satellite) into ' +
    'two detectors for aircraft in flight, one looking down from orbit and one for cameras looking at aircraft, then ' +
    'building real test sets for both and training the sky model longer on more data. All numbers come from files in ' +
    'the repository (docs/report.md, data/aot_part2/results, data/s2_flying).');
}

// ================= summary =================
{
  const s = content('Where we ended up', 'Both detectors now have real test sets large enough to trust',
    'Headline numbers. Sky: AOT part 2 has 1,813 crops with 1,708 aircraft from 1,312 flights that no model trained on. Average precision (AP) uses centre-distance ' +
    'matching (a hit is within max(8 px, half the box) of an aircraft), because AOT aircraft are often under 10 px. ' +
    'Satellite: data/s2_flying is a label-guided Sentinel-2 set of 927 aircraft at 10 European airports, plus 2,134 ' +
    'background chips; S2Net is trained on it and tested on data/s2_airborne (19 aircraft, 2025). ' +
    'background chips. The two test-set factors compare with the previous test sets: 260 aircraft (AOT part 1 ' +
    'validation) and 19 aircraft (data/s2_airborne).');
  const w = (CW - 0.9) / 4;
  const y = TOP + 0.35;
  const s50 = p2.yolo11s_3frame_p13_e50;
  stat(s, M, y, w, 3.0, s50.ap.toFixed(2), 'Sky detector average precision on 1,708 aircraft from flights it never saw ' +
    '(untrained COCO model: ' + p2.coco_yolo11n.ap.toFixed(2) + ')');
  stat(s, M + (w + 0.3), y, w, 3.0, D.matched_fpr.S2Net['0.05'] + ' of 19', 'aircraft found by the new S2Net in an ' +
    'independent Sentinel-2 set, at 5% clutter false alarms (PlaneNet: ' + D.matched_fpr.PlaneNet['0.05'] + ')', { bigSize: 44 });
  stat(s, M + 2 * (w + 0.3), y, w, 3.0, '6.6× | 49×', 'Larger real test sets: sky 260 → 1,708 aircraft; ' +
    'satellite 19 → 927 aircraft', { bigSize: 40 });
  stat(s, M + 3 * (w + 0.3), y, w, 3.0, '6.5×', 'More training data (1,548 → 10,051 crops), 50 epochs on the Mac\'s GPU');
  caption(s, 'Average precision: area under the precision-recall curve, matching by centre distance. False alarms: ' +
    'share of 2,134 random background chips scored above 0.5.', M, y + 3.35, CW, 0.6, { color: HEX.muted });
}

// ================= agenda =================
{
  const s = content('Contents', 'Seven parts, from the starting point to what to do next',
    'Sections follow the order of the work. The appendix-style slides at the end (repo fixes, reproducibility) ' +
    'document what changed in the repository and how to rerun everything.');
  const items = [
    ['The problem', 'What the original repo did, and the two viewpoints'],
    ['Data', 'AOT video, PlanesNet, Sentinel-2; splits and leakage'],
    ['Models', 'PlaneNet and the YOLO variants'],
    ['Training', '12 epochs on CPU, then 50 epochs on the Apple GPU'],
    ['Sky results', 'Validation, a 6.6× larger test set, curves, examples'],
    ['Satellite results', 'Building a 927-aircraft Sentinel-2 test set'],
    ['Repository and next steps', 'Fixes, reproducibility, limitations'],
  ];
  const colW = (CW - 0.5) / 2;
  items.forEach(([t, d], i) => {
    const col = i < 4 ? 0 : 1;
    const row = i < 4 ? i : i - 4;
    const x = M + col * (colW + 0.5);
    const y = TOP + 0.25 + row * 1.15;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.05, w: 0.7, h: 0.7, fill: { color: C.accent2 }, line: { type: 'none' },
      objectName: 'Agenda badge ' + (i + 1) });
    s.addText(String(i + 1), { x, y: y + 0.05, w: 0.7, h: 0.7, align: 'center', valign: 'middle', fontSize: 22, bold: true,
      color: C.background1, fontFace: B.THEME.headFontFace, isTextBox: true, margin: 0, objectName: 'Agenda number' });
    s.addText([{ text: t, options: { bold: true, fontSize: 20, breakLine: true } },
      { text: d, options: { fontSize: 14, color: HEX.muted } }],
    { x: x + 0.95, y, w: colW - 1.0, h: 0.85, valign: 'middle', color: C.text1, isTextBox: true, margin: 0,
      objectName: 'Agenda item' });
  });
}

// ================= 1. the problem =================
divider('The problem', 'The repo classified parked aircraft; we needed aircraft in flight',
  'The original repository (Bob Hammell, 2017-2020) trains a TFLearn CNN on PlanesNet and slides it across Planet ' +
  'scenes. PlanesNet is almost entirely parked and taxiing aircraft at Californian airports.');
{
  const s = content('Starting point', 'Data, model and detector all assumed aircraft parked on concrete',
    'Original pipeline: model.py (TFLearn CNN, 877,058 parameters, 93% in one dense layer), train.py (PlanesNet JSON), ' +
    'detector.py (20 px window at 2 px steps, one model call per window, every hit boxed at 20x20). TensorFlow 1 and ' +
    'TFLearn no longer install on current Python, so the files are kept only for reference.');
  table(s, ['Part', 'Original', 'Problem for aircraft in flight'], [
    ['Data', 'PlanesNet: 32,000 RGB chips, 20×20 px, Planet imagery at about 3 m per pixel',
      'Almost every aircraft is parked or taxiing; no cloud, sea or haze backgrounds'],
    ['Model', 'TFLearn CNN: 2 conv blocks and a 512-unit dense layer, normalised with airport statistics',
      'Learns "white cross on grey concrete"; the normalisation is tied to tarmac brightness'],
    ['Detector', '20 px window slid in 2 px steps: about 280,000 model calls per scene',
      'Very slow, no score threshold, no use of motion'],
    ['Stack', 'TensorFlow 1, TFLearn, Python 3.5', 'No longer installs on current Python'],
  ], M, TOP + 0.2, CW, [1.5, 5.1, 5.53], { size: 15, rowH: [0.45, 0.85, 0.85, 0.7, 0.55] });
}
{
  const s = content('Two viewpoints, two pipelines', '"Aircraft in flight" means looking down from orbit or up from a camera',
    'satellite/ is PyTorch: PlaneNet, a fully convolutional classifier trained on PlanesNet plus synthetic in-flight ' +
    'samples, with static-object suppression across dates and band-offset motion cues. sky/ fine-tunes Ultralytics ' +
    'YOLO on Amazon\'s Airborne Object Tracking (AOT) dataset, with tiled inference, multi-frame tracking and a ' +
    '3-frame temporal input.');
  const colW = (CW - 0.5) / 2;
  s.addText('Satellite: looking down', { x: M, y: TOP, w: colW, h: 0.45, fontSize: 20, bold: true, color: C.text1,
    isTextBox: true, margin: 0, objectName: 'Column title' });
  let y = picture(s, fig('fig_s2_examples.png'), M, TOP + 0.55, colW, 1.857, 'Sentinel-2 aircraft with colour copies');
  bullets(s, ['Planet and Sentinel-2 RGB, 3 to 10 m per pixel', 'Aircraft are a few pixels; motion splits them into ' +
    'blue, green and red copies', 'Code: satellite/ (PlaneNet)'], M, y + 0.2, colW, 1.3, { size: 14 });
  const x2 = M + colW + 0.5;
  s.addText('Sky: cameras looking at aircraft', { x: x2, y: TOP, w: colW, h: 0.45, fontSize: 20, bold: true, color: C.text1,
    isTextBox: true, margin: 0, objectName: 'Column title' });
  y = picture(s, fig('fig_temporal_stack.png'), x2, TOP + 0.55, colW, 3.589, 'Helicopter in three frames and stacked');
  bullets(s, ['AOT: 2448×2048 greyscale video from aircraft-mounted cameras, about 10 frames per second',
    'Most aircraft are under 20 px, against sky or ground clutter', 'Code: sky/ (YOLO11 fine-tuning)'],
  x2, y + 0.2, colW, 2.2, { size: 14 });
}

// ================= 2. data =================
divider('Data', 'Real airborne video for the sky, PlanesNet plus synthetic samples and real Sentinel-2 for the satellite',
  'Every dataset used, how it was split, and the leakage checks.');
{
  const s = content('Datasets', 'Each pipeline is trained on one source and tested on real aircraft in flight',
    'AOT: s3://airborne-obj-detection-challenge-training, CDLA-Permissive 1.0, about 4,943 flights in three parts. ' +
    'PlanesNet: Kaggle rhammell/planesnet, supplied on the data branch because Kaggle was blocked in the cloud. ' +
    'Sentinel-2: L2A COGs in the public sentinel-cogs bucket, found through Earth Search. Flying Airplanes labels: ' +
    'Pamplona Segundo et al. 2021, github.com/maups/covid19-custom-script-contest, MIT licence.');
  const c = D.crops;
  table(s, ['Dataset', 'Used for', 'Amount used', 'Access'], [
    ['AOT (Amazon Prime Air), parts 1-3', 'Sky training, validation and test',
      `${(c.p1_train + c.p3_train).toLocaleString('en-US')} training crops, ${c.p1_val} validation, ` +
      `${c.part2_test.toLocaleString('en-US')} test (part 2)`, 'Public S3, no account'],
    ['PlanesNet', 'Satellite training: aircraft seen from above', '32,000 chips (8,000 aircraft)', 'Kaggle; data branch'],
    ['Sentinel-2 L2A (10 m)', 'Satellite test sets', '19 aircraft (s2_airborne), 927 aircraft (s2_flying)',
      'AWS COGs via Earth Search'],
    ['Flying Airplanes labels', 'Guide for the 927-aircraft test set', '1,782 labels at 15 European airports',
      'GitHub, MIT licence'],
    ['COCO yolo11n / yolo11s', 'Starting weights and baseline', 'Pretrained "airplane" class', 'Ultralytics releases'],
  ], M, TOP + 0.2, CW, [3.0, 3.1, 3.8, 2.23], { size: 14 });
}
{
  const s = content('AOT crops for the sky detector', 'Native-resolution crops keep 6 px aircraft visible to YOLO',
    'Full 2448x2048 frames would shrink a 6 px aircraft below one pixel at YOLO\'s 640 px input. sky/prepare_aot.py ' +
    'maps AOT instance names to 5 classes (airplane, helicopter, bird, drone, unknown); birds and unknowns stay ' +
    'labelled but are ignored in scoring. About 28% of part 1 frames listed in groundtruth.csv are missing from the ' +
    'bucket (HTTP 404); the script over-plans by 35% and now retries transient network errors.');
  const steps = [
    ['Map classes', 'Airplane, helicopter, drone are scored; birds and unknown objects are ignored'],
    ['Pick frames', 'At most 6 per flight, 20+ frames apart, with neighbours t−2 and t+2'],
    ['Crop 640 px', 'At native resolution around an aircraft; about 15% empty crops'],
    ['Stack 3 frames', 't−2, t, t+2 aligned by phase correlation, stored as colour channels'],
    ['Split by flight', 'md5 of the flight id: 20% of flights held out, never split by frame'],
  ];
  const bw = (CW - 4 * 0.3) / 5;
  steps.forEach(([t, d], i) => {
    const x = M + i * (bw + 0.3);
    card(s, x, TOP + 0.3, bw, 2.6, { name: 'Step card' });
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: TOP + 0.5, w: 0.55, h: 0.55, fill: { color: C.accent1 },
      line: { type: 'none' }, objectName: 'Step badge' });
    s.addText(String(i + 1), { x: x + 0.2, y: TOP + 0.5, w: 0.55, h: 0.55, align: 'center', valign: 'middle', fontSize: 16,
      bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: 'Step number' });
    s.addText([{ text: t, options: { bold: true, fontSize: 16, breakLine: true } },
      { text: d, options: { fontSize: 13 } }], { x: x + 0.2, y: TOP + 1.2, w: bw - 0.4, h: 1.6, valign: 'top',
      color: C.text1, isTextBox: true, margin: 0, objectName: 'Step text' });
  });
  const sw = (CW - 2 * 0.3) / 3;
  [['1.8 M', 'label rows in part 1 alone'], ['< 10 px', 'size of 43% of the test aircraft'],
    ['~28%', 'of part 1 frames missing from the bucket']].forEach(([b, l], i) => {
    s.addText([{ text: b + '  ', options: { fontSize: 30, bold: true, color: C.accent2, fontFace: B.THEME.headFontFace } },
      { text: l, options: { fontSize: 15, color: C.text1 } }], { x: M + i * (sw + 0.3), y: TOP + 3.3, w: sw, h: 0.8,
      valign: 'middle', isTextBox: true, margin: 0, objectName: 'Fact' });
  });
}
{
  const s = content('AOT parts share flights', 'Part 2 is a clean test set; part 3 is used for training without part 2\'s flights',
    'Flight ids compared across the three groundtruth.csv files. Part 1 and part 2 share none, so part 2 is a clean ' +
    'test set for every model trained on part 1. Part 3 shares 352 flights with part 1 and 364 with part 2, so ' +
    'training on part 3 as-is would leak into the test set. prepare_aot --exclude-flights part2.csv removes them; ' +
    'keeping --val-fraction 0.2 puts any flight shared by parts 1 and 3 on the same side (train or validation) in ' +
    'both, because the split is a hash of the flight id. Checked after building: 0 training flights in part 2, 0 ' +
    'training flights in validation.');
  const parts = [['Part 1', '1,311 flights', 'Training (80%) and validation (20%)', C.accent1],
    ['Part 3', '1,531 flights', 'Training only, minus part 2\'s flights', C.accent3],
    ['Part 2', '1,312 flights', 'Test set: 1,708 aircraft', C.accent2]];
  const bw = 3.3;
  const xs = [M + 0.2, M + (CW - bw) / 2, M + CW - bw - 0.2];
  parts.forEach(([t, n, d, col], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xs[i], y: TOP + 0.4, w: bw, h: 1.9, rectRadius: 0.1,
      fill: { color: C.background2 }, line: { color: col, width: 2.5 }, objectName: t + ' box' });
    s.addText([{ text: t, options: { bold: true, fontSize: 22, color: col, breakLine: true } },
      { text: n, options: { fontSize: 16, bold: true, color: C.text1, breakLine: true } },
      { text: d, options: { fontSize: 14, color: C.text1 } }], { x: xs[i] + 0.25, y: TOP + 0.5, w: bw - 0.5, h: 1.7,
      valign: 'middle', isTextBox: true, margin: 0, objectName: t + ' text' });
  });
  const links = [[0, 1, '352 shared', 'Same side in both (flight hash)'], [1, 2, '364 shared', 'Removed from training'],
    [0, 2, '0 shared', 'Clean test set']];
  links.forEach(([a, b, n, d], i) => {
    const x = i < 2 ? xs[a] + bw + 0.05 : M + 0.2;
    const w = i < 2 ? xs[b] - xs[a] - bw - 0.1 : CW - 0.4;
    const y = i < 2 ? TOP + 0.9 : TOP + 2.75;
    if (i < 2) {
      s.addShape(pres.shapes.LINE, { x, y: TOP + 1.35, w, h: 0, line: { color: HEX.muted, width: 1.5, dashType: 'dash' },
        objectName: 'Overlap link' });
      s.addText([{ text: n, options: { bold: true, fontSize: 15, breakLine: true } }, { text: d, options: { fontSize: 12 } }],
        { x: x - 0.1, y: TOP + 1.45, w: w + 0.2, h: 0.75, align: 'center', valign: 'top', color: C.text1, isTextBox: true,
          margin: 0, objectName: 'Overlap label' });
    } else {
      card(s, x, y, w, 1.0, { name: 'Part 1 and 2 card' });
      s.addText([{ text: 'Part 1 and part 2: ' + n + '. ', options: { bold: true } },
        { text: 'Every model trained on part 1 can be tested on part 2. Built with prepare_aot --part part2 ' +
          '--train 0 --val-fraction 1: 1,813 crops.' }], { x: x + 0.3, y, w: w - 0.6, h: 1.0, valign: 'middle', fontSize: 15,
        color: C.text1, isTextBox: true, margin: 0, objectName: 'Part 1 and 2 text' });
    }
  });
  caption(s, 'Checked after building: no training flight is in validation or in part 2.', M + 0.2, TOP + 4.05, CW - 0.4, 0.4,
    { color: HEX.muted });
}
{
  const s = content('Training data grew 6.5 times', 'From 1,548 crops of part 1 to 10,051 crops from parts 1 and 3',
    'The 12-epoch models (report runs on a 4-core cloud CPU) used 1,548 training crops from part 1. The 50-epoch run ' +
    'used 5,874 crops from part 1\'s training flights and 4,177 from part 3 with part 2 excluded. On a slow ' +
    'connection about a quarter of downloads first failed with connection resets and timeouts; after adding retries ' +
    'with backoff, part 3 built with 0 failures.');
  const c = D.crops;
  s.addChart(pres.charts.BAR, [
    { name: 'Part 1 crops', labels: ['12-epoch models', '50-epoch model'], values: [1548, c.p1_train] },
    { name: 'Part 3 crops', labels: ['12-epoch models', '50-epoch model'], values: [0, c.p3_train] },
  ], chartOpts({ x: M, y: TOP + 0.1, w: 7.4, h: 4.8, barDir: 'bar', barGrouping: 'stacked', barGapWidthPct: 60,
    chartColors: [T.accent1, T.accent3], showValue: true, dataLabelPosition: 'ctr', dataLabelColor: 'FFFFFF',
    dataLabelFormatCode: '#,##0;;;', showLegend: true, legendPos: 'b', title: 'Training crops',
    valAxisLabelFormatCode: '#,##0', valAxisMaxVal: 11000, catAxisLabelFontSize: 14 }));
  const x = M + 7.8;
  const w = CW - 7.8;
  stat(s, x, TOP + 0.2, w, 1.55, c.p1_val + ' crops', 'validation: 230 held-out part 1 flights', { bigSize: 32 });
  stat(s, x, TOP + 1.95, w, 1.55, '1,813 crops', 'test (part 2): 1,708 aircraft, 1,312 flights', { bigSize: 32 });
  stat(s, x, TOP + 3.7, w, 1.2, '0 failures', 'building part 3 after adding download retries', { bigSize: 32,
    bigColor: C.accent3 });
}
{
  const s = content('Three frames as one image', 'Static scenery stacks to grey; anything moving leaves coloured fringes',
    'AOT is greyscale, so the three colour channels are free. sky/temporal.py aligns frames t-2 and t+2 to frame t by ' +
    'phase correlation (to cancel the camera\'s own motion), matches brightness, and stacks them as (B, G, R) = ' +
    '(t-k, t, t+k). Hue and saturation augmentation is switched off for these models because it would scramble the ' +
    'time channels. No layer changes are needed.');
  const y = picture(s, fig('fig_temporal_stack.png'), M + 0.6, TOP + 0.2, CW - 1.2, 3.589,
    'Helicopter in frames t-2, t, t+2 and stacked');
  bullets(s, [
    [{ text: 'Align: ', options: { bold: true } }, { text: 'neighbours are shifted onto frame t by phase correlation' }],
    [{ text: 'Stack: ', options: { bold: true } }, { text: 'frames t−2, t and t+2 become the three colour channels' }],
    [{ text: 'Effect: ', options: { bold: true } }, { text: 'at equal recall, about 40% fewer false detections than one frame' }],
  ], M + 0.6, y + 0.35, CW - 1.2, 1.5, { size: 16 });
}
{
  const s = content('Satellite training data', 'PlanesNet aircraft moved into the air, plus hard negatives',
    'satellite/augment.py cuts each PlanesNet aircraft out of its chip (plane_mask) with a threshold scaled to each ' +
    'chip\'s background variation; 7,489 of 8,000 aircraft are kept with clean outlines. They are pasted onto ' +
    'procedural cloud, sea with glint, farmland, haze and contrails, with simulated band-offset ghosting. 30% of ' +
    'synthetic negatives are blobs of real ground texture, so "tarmac in the sky" is not a shortcut.');
  const w = (CW - 0.4) / 2;
  let y = picture(s, fig('fig_masks.png'), M, TOP + 0.1, w, 1.897, 'Aircraft masks under three settings');
  caption(s, 'Mask settings: version 3 keeps the silhouette and 94% of aircraft', M, y + 0.1, w, 0.6);
  y = picture(s, fig('fig_synthetic.png'), M + w + 0.4, TOP + 0.1, w, 1.897, 'Synthetic in-flight samples and negatives');
  caption(s, 'Rows: real PlanesNet, synthetic in-flight positives, tarmac distractors, hard negatives', M + w + 0.4,
    y + 0.1, w, 0.6);
}

// ================= 3. models =================
divider('Models', 'A small fully convolutional satellite classifier, and YOLO11 variants for the sky',
  'Architecture changes for both pipelines.');
{
  const s = content('PlaneNet', 'Nine times fewer parameters, and one pass over a whole scene',
    'Two 3x3 conv blocks (32 then 64 channels, batch norm, ReLU, 2x2 pooling), then the old dense layer rewritten as a ' +
    '2x2 convolution to 128 channels, dropout and a 1x1 convolution to one score. No convolution is padded and the ' +
    'pools are aligned, so on a full scene the network outputs one score every 4 px, identical to classifying each ' +
    '20x20 window separately; tests/test_satellite.py checks this.');
  const y = picture(s, fig('fig_architecture.png'), M, TOP + 0.05, 7.4, 1.517, 'Original network versus PlaneNet');
  const x = M + 7.8;
  const w = CW - 7.8;
  stat(s, x, TOP + 0.05, w, 1.55, '877k → 99k', 'parameters; the dense layer becomes a 2×2 convolution', { bigSize: 34 });
  stat(s, x, TOP + 1.8, w, 1.55, '1 pass', 'per scene instead of about 280,000 window calls', { bigSize: 34 });
  stat(s, x, TOP + 3.55, w, 1.3, '~4 s', 'for a 1225×1139 scene on CPU', { bigSize: 34 });
}
{
  const s = content('Sky detector variants', 'Same recipe for all; they differ in input, detection layers and size',
    'yolo11-p2 adds a fourth detection output at 4 px per cell for 4-10 px targets; 297 of 593 weight tensors transfer ' +
    'from yolo11n, the rest start untrained. Its config was lost to a .gitignore rule and rebuilt from the trained ' +
    'checkpoint, verified tensor for tensor. yolo11s has the same structure with twice the channels.');
  table(s, ['Model', 'Parameters', 'Input', 'Detection layers', 'Training'], [
    ['yolo11n, COCO weights (baseline)', '2.62 M', '1 frame', '8, 16, 32 px', 'None'],
    ['yolo11n fine-tuned', '2.62 M', '1 frame', '8, 16, 32 px', '12 epochs, 1,548 crops, CPU'],
    ['yolo11n fine-tuned', '2.62 M', '3 frames', '8, 16, 32 px', '12 epochs, 1,548 crops, CPU'],
    ['yolo11n-p2', '2.74 M', '3 frames', '4, 8, 16, 32 px', '12 epochs, 1,548 crops, CPU'],
    ['yolo11s', '9.46 M', '3 frames', '8, 16, 32 px', '12 epochs, 1,548 crops, CPU'],
    [{ text: 'yolo11s', options: { bold: true, fontSize: 15, color: C.text1, fill: { color: C.background2 } } },
      { text: '9.46 M', options: { bold: true, fontSize: 15, color: C.text1, fill: { color: C.background2 } } },
      { text: '3 frames', options: { bold: true, fontSize: 15, color: C.text1, fill: { color: C.background2 } } },
      { text: '8, 16, 32 px', options: { bold: true, fontSize: 15, color: C.text1, fill: { color: C.background2 } } },
      { text: '50 epochs, 10,051 crops, Apple GPU', options: { bold: true, fontSize: 15, color: C.accent2,
        fill: { color: C.background2 } } }],
  ], M, TOP + 0.2, CW, [3.4, 1.6, 1.5, 2.3, 3.33], { size: 15 });
}

// ================= 4. training =================
divider('Training', '12 epochs on a 4-core CPU, then 50 epochs on the Apple M5 Pro GPU',
  'Recipe: 640 px, batch 16, COCO weights, centre-distance scoring. Loss: box (CIoU) + classification (BCE) + DFL.');
{
  const s = content('12-epoch runs on CPU', 'Every model was still improving when training stopped',
    'From the cloud runs in docs/report.md: 12 epochs at 640 px, batch 16, about 6 minutes per epoch for yolo11n on 4 ' +
    'cores. Validation loss more than halved and mAP50 roughly quintupled, still rising at the last epoch; the 1-frame ' +
    'and 3-frame models swapped the lead several times. These mAP50 values count all five classes with box-overlap ' +
    'matching, so they run lower than the aircraft-only average precision used elsewhere.');
  picture(s, fig('fig_loss_curves.png'), M + 0.4, TOP + 0.1, CW - 0.8, 2.211, 'Loss curves of 12-epoch runs');
}
{
  const s = content('50-epoch run on the Apple GPU', 'Validation mAP50 rose from 0.21 at epoch 12 to 0.31 at epoch 45',
    'yolo11s, 3-frame input, 10,051 training crops, 540 validation crops, --device mps on an Apple M5 Pro (24 GB). ' +
    'About 6.7 minutes per epoch (the 12-epoch yolo11n run took about 6 minutes per epoch on 1,548 crops on a 4-core ' +
    'CPU). Mosaic off for the last 10 epochs. The best epoch by Ultralytics fitness (0.1 mAP50 + 0.9 mAP50-95) is 45; ' +
    'that checkpoint is models/sky_yolo11s_3frame_e50.pt. The validation set differs from the 12-epoch runs\' (540 vs ' +
    '287 crops), so these mAP50 values are not comparable with the previous slide.');
  const t = D.train50;
  const ep = t.epoch.map((e) => String(Math.round(e)));
  const w = (CW - 0.4) / 2;
  s.addChart(pres.charts.LINE, [{ name: 'Validation mAP50', labels: ep, values: t['metrics/mAP50(B)'] }],
    chartOpts({ x: M, y: TOP + 0.1, w, h: 4.8, chartColors: [T.accent2], lineSize: 2, lineDataSymbol: 'none',
      title: 'Validation mAP50 (all classes)', valAxisMinVal: 0, valAxisMaxVal: 0.35, valAxisLabelFormatCode: '0.00',
      catAxisLabelFrequency: 5, catAxisTitle: 'Epoch', showCatAxisTitle: true }));
  s.addChart(pres.charts.LINE, [
    { name: 'Training', labels: ep, values: t['train/box_loss'] },
    { name: 'Validation', labels: ep, values: t['val/box_loss'] },
  ], chartOpts({ x: M + w + 0.4, y: TOP + 0.1, w, h: 4.8, chartColors: [T.accent1, T.accent2], lineSize: 2,
    lineDataSymbol: 'none', title: 'Box loss', showLegend: true, legendPos: 'b', valAxisLabelFormatCode: '0.0',
    valAxisMinVal: 1, catAxisLabelFrequency: 5, catAxisTitle: 'Epoch', showCatAxisTitle: true }));
}

require('./slides2.js');
