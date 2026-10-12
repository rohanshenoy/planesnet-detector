const B = require('./build.js');
const { pres, C, T, HEX, MODEL, D, PR, fig, img, M, CW, TOP, BOTTOM, W, content, divider, card, stat, bullets, caption,
  picture, chartOpts, table, pct, p2 } = B;

// ================= 5. sky results =================
divider('Sky results', 'Validation, a 6.6× larger test set, precision-recall curves and examples',
  'Scoring: a prediction is a hit when its centre lies within max(8 px, half the box) of an airplane, helicopter or ' +
  'drone. Birds and unidentified objects are ignored. Precision and recall are read at the threshold with the best F1.');
{
  const s = content('Results hold on a 6.6× larger test set', 'Every model scores within 0.01 of its validation result',
    'Validation: 287 crops, 260 aircraft from held-out part 1 flights (docs/report.md). Test: AOT part 2, 1,813 crops, ' +
    '1,708 aircraft from 1,312 flights, none shared with part 1. Average precision: COCO 0.11 vs 0.12, yolo11n ' +
    '1-frame 0.55 vs 0.56, 3-frame 0.58 vs 0.58, yolo11s 0.63 vs 0.64, yolo11n-p2 0.41 vs 0.42. The 50-epoch model ' +
    'used a different validation set, so it is not in this chart.');
  const names = ['coco_yolo11n', 'yolo11n_1frame', 'yolo11n_3frame', 'yolo11s_3frame', 'yolo11n-p2_3frame'];
  const val = { coco_yolo11n: 0.11, yolo11n_1frame: 0.55, yolo11n_3frame: 0.58, yolo11s_3frame: 0.63, 'yolo11n-p2_3frame': 0.41 };
  const labels = names.map((n) => MODEL[n].short);
  s.addChart(pres.charts.BAR, [
    { name: 'Validation (260 aircraft, part 1)', labels, values: names.map((n) => val[n]) },
    { name: 'Test (1,708 aircraft, part 2)', labels, values: names.map((n) => +p2[n].ap.toFixed(2)) },
  ], chartOpts({ x: M, y: TOP + 0.05, w: CW, h: 4.85, barDir: 'col', barGapWidthPct: 70, chartColors: ['86B6EF', T.accent1],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', showLegend: true, legendPos: 'b',
    title: 'Average precision', valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0.0', catAxisLabelFontSize: 13 }));
}
{
  const s = content('Average precision on AOT part 2', 'Training longer on more data lifted it from 0.64 to 0.80',
    'All six models on the same 1,708 test aircraft. Values: COCO yolo11n 0.12, yolo11n-p2 0.42, yolo11n 1 frame 0.56, ' +
    'yolo11n 3 frames 0.58, yolo11s 12 epochs 0.64, yolo11s 50 epochs 0.80 (0.8047). The P2 model was undertrained at ' +
    '12 epochs: about half its weights started untrained.');
  const names = ['coco_yolo11n', 'yolo11n-p2_3frame', 'yolo11n_1frame', 'yolo11n_3frame', 'yolo11s_3frame',
    'yolo11s_3frame_p13_e50'];
  s.addChart(pres.charts.BAR, [{ name: 'Average precision', labels: names.map((n) => MODEL[n].label),
    values: names.map((n) => +p2[n].ap.toFixed(2)) }],
  chartOpts({ x: M, y: TOP + 0.05, w: 8.2, h: 4.85, barDir: 'bar', barGapWidthPct: 45, showTitle: false,
    chartColors: names.map((n) => MODEL[n].color), showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00',
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0.0', catAxisLabelFontSize: 13, dataLabelFontSize: 13 }));
  const x = M + 8.6;
  const w = CW - 8.6;
  const s50 = p2.yolo11s_3frame_p13_e50;
  const s12 = p2.yolo11s_3frame;
  stat(s, x, TOP + 0.1, w, 1.55, pct(s50.precision) + ' / ' + pct(s50.recall), 'precision / recall at best F1 (12 epochs: ' +
    pct(s12.precision) + ' / ' + pct(s12.recall) + ')', { bigSize: 34 });
  stat(s, x, TOP + 1.85, w, 1.55, pct(s50.max_recall), 'maximum recall at the lowest threshold (12 epochs: ' +
    pct(s12.max_recall) + ')', { bigSize: 34 });
  stat(s, x, TOP + 3.6, w, 1.3, '354', 'test aircraft found by 50 epochs and missed by 12', { bigSize: 34 });
}
{
  const s = content('Precision against recall', 'At 70% recall the 50-epoch model keeps 92% precision; the others are below 50%',
    'Interpolated precision (upper envelope) against recall on AOT part 2, sampled every 1% of recall; each curve ends ' +
    'at the model\'s maximum recall. Precision at 50% recall: yolo11s 50 epochs 99%, 12 epochs 83%, yolo11n 3 frames ' +
    '75%, 1 frame 67%. At 70% recall: 92%, 47%, 36%, 35%. The COCO model never reaches 20% recall (maximum 15%).');
  const names = ['coco_yolo11n', 'yolo11n_1frame', 'yolo11n_3frame', 'yolo11s_3frame', 'yolo11s_3frame_p13_e50'];
  const grid = PR.yolo11s_3frame_p13_e50.recall_grid;
  const labels = grid.map((g) => Math.round(g * 100) + '%');
  s.addChart(pres.charts.LINE, names.map((n) => ({ name: MODEL[n].label, labels, values: PR[n].precision })),
    chartOpts({ x: M, y: TOP + 0.05, w: CW, h: 4.85, chartColors: names.map((n) => MODEL[n].color), lineSize: 2.5,
      lineDataSymbol: 'none', showTitle: false, showLegend: true, legendPos: 'r', valAxisMinVal: 0, valAxisMaxVal: 1,
      valAxisLabelFormatCode: '0%', catAxisLabelFrequency: 10, catAxisTitle: 'Recall', showCatAxisTitle: true,
      valAxisTitle: 'Precision', showValAxisTitle: true, objectName: 'PR chart' }));
}
{
  const s = content('Where the gain came from', 'Largest gains where the model was weakest: ground backgrounds and tiny aircraft',
    'Recall at each model\'s best-F1 threshold, by group, on AOT part 2. Groups: 1,095 aircraft against sky, 590 ' +
    'against ground (23 unlabelled), 727 under 10 px, 981 at 10 px or more. Data and epochs changed together in the ' +
    '50-epoch run, so this run cannot say how much of the gain comes from each.');
  const groups = [['above_horizon (sky background)', 'Against sky (1,095)'], ['below_horizon (ground background)',
    'Against ground (590)'], ['tiny (<10 px)', 'Under 10 px (727)'], ['small+ (>=10 px)', '10 px or more (981)']];
  const labels = groups.map((g) => g[1]);
  s.addChart(pres.charts.BAR, ['yolo11s_3frame', 'yolo11s_3frame_p13_e50'].map((n) => ({ name: MODEL[n].label, labels,
    values: groups.map((g) => p2[n].recall_by_group[g[0]][0]) })),
  chartOpts({ x: M, y: TOP + 0.05, w: CW, h: 4.85, barDir: 'col', barGapWidthPct: 70,
    chartColors: [T.accent1, T.accent2], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0%',
    showLegend: true, legendPos: 'b', title: 'Recall at best F1, by group', valAxisMinVal: 0, valAxisMaxVal: 1,
    valAxisLabelFormatCode: '0%', catAxisLabelFontSize: 14 }));
}
if (p2.yolo11s_3frame_p1_e50) {
  const s = content('Data or epochs?', 'Part 1 alone reaches the same ' + p2.yolo11s_3frame_p1_e50.ap.toFixed(3) +
    ' as parts 1 + 3: the part 3 data added nothing measurable',
    'Three yolo11s 3-frame runs scored on AOT part 2. 12 epochs on 1,548 part 1 crops (cloud CPU); 50 epochs on 5,874 ' +
    'part 1 crops; 50 epochs on 10,051 crops from parts 1 and 3. The last two differ only in the part 3 data, at equal ' +
    'epochs. The first two differ in both epochs and amount of part 1 data, so a run of 50 epochs on 1,548 crops is ' +
    'still needed to isolate epochs. Validation thresholds and recipe as before; one run each.');
  const runs = [['yolo11s_3frame', '12 epochs, 1,548 crops (part 1)', T.accent1],
    ['yolo11s_3frame_p1_e50', '50 epochs, 5,874 crops (part 1)', '86B6EF'],
    ['yolo11s_3frame_p13_e50', '50 epochs, 10,051 crops (parts 1 + 3)', T.accent2]];
  s.addChart(pres.charts.BAR, [{ name: 'Average precision', labels: runs.map((r) => r[1]),
    values: runs.map((r) => +p2[r[0]].ap.toFixed(3)) }],
  chartOpts({ x: M, y: TOP + 0.05, w: 7.4, h: 4.85, barDir: 'bar', barGapWidthPct: 45, chartColors: runs.map((r) => r[2]),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.000', title: 'Average precision on AOT part 2',
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0.0', catAxisOrientation: 'maxMin', catAxisLabelFontSize: 13 }));
  const x = M + 7.8;
  const w = CW - 7.8;
  const g = (n, k) => pct(p2[n].recall_by_group[k][0]);
  table(s, ['Recall at best F1', 'Part 1', 'Parts 1 + 3'], [
    ['Against sky', g('yolo11s_3frame_p1_e50', 'above_horizon (sky background)'), g('yolo11s_3frame_p13_e50', 'above_horizon (sky background)')],
    ['Against ground', g('yolo11s_3frame_p1_e50', 'below_horizon (ground background)'), g('yolo11s_3frame_p13_e50', 'below_horizon (ground background)')],
    ['Under 10 px', g('yolo11s_3frame_p1_e50', 'tiny (<10 px)'), g('yolo11s_3frame_p13_e50', 'tiny (<10 px)')],
    ['Precision / recall', pct(p2.yolo11s_3frame_p1_e50.precision) + ' / ' + pct(p2.yolo11s_3frame_p1_e50.recall),
      pct(p2.yolo11s_3frame_p13_e50.precision) + ' / ' + pct(p2.yolo11s_3frame_p13_e50.recall)],
  ], x, TOP + 0.1, w, [2.2, 1.2, w - 3.4], { size: 14 });
  caption(s, 'Equal epochs, so the two 50-epoch runs differ only in the part 3 data. The gain from 0.64 comes from ' +
    '50 epochs and 3.8× more part 1 crops together.', x, TOP + 3.2, w, 1.2, { color: HEX.muted });
}
{
  const s = content('What the longer run finds', 'Aircraft the 12-epoch model missed: against ground, and 6 px against sky',
    'Four of the 354 part-2 aircraft that the 50-epoch model finds at its best-F1 threshold (' +
    PR.yolo11s_3frame_p13_e50.threshold.toFixed(2) + ') and the 12-epoch model misses at its own (' +
    PR.yolo11s_3frame.threshold.toFixed(2) + '). Green circle: labelled aircraft. Orange boxes: detections above each model\'s ' +
    'threshold, drawn on the greyscale frame (the models see the 3-frame stack). In the second column the 50-epoch ' +
    'model also draws a duplicate box, which the scoring counts as a false alarm. Windows are 160 px of a 640 px crop.');
  picture(s, img('sky_before_after.png'), M + 1.4, TOP - 0.05, CW - 2.8, 1.875, 'Before and after examples on AOT part 2');
}
{
  const s = content('Full frames and tracking', 'On full frames the 50-epoch model still leads, but by far less than on crops',
    '8 clips of 50 full 2448x2048 frames from held-out part 1 flights, 400 aircraft instances, tiled at native resolution ' +
    '(sky/evaluate_clips.py). Thresholds: each model\'s best-F1 threshold on the 540 held-out part 1 validation crops. ' +
    'Full-frame average precision counts every detection at confidence 0.05 or more, so it does not depend on a ' +
    'threshold. Raw: every detection above threshold. Offline: every detection of a track confirmed at any frame (5 hits). ' +
    'Crops are cut around aircraft (about 15% empty), so crop scores reward finding aircraft more than ignoring a whole ' +
    'frame of clutter: the 12-epoch yolo11s raises 0.97 false alarms per frame. 400 instances from 8 clips are about 8 ' +
    'tracked aircraft, so differences of a few points are not significant.');
  const names = [['n_1frame', 'yolo11n_1frame'], ['n_3frame', 'yolo11n_3frame'], ['s_12ep', 'yolo11s_3frame'],
    ['s_50ep', 'yolo11s_3frame_p13_e50']];
  const R = D.clips.results;
  table(s, ['Model', 'Full-frame AP', 'Raw: found / precision', 'False alarms per frame', 'Offline: found / precision'],
    names.map(([k, m]) => [MODEL[m].label, D.clips_ap[k].ap.toFixed(2), pct(R[k].raw.recall) + ' / ' + pct(R[k].raw.precision),
      R[k].raw.false_alarms_per_frame.toFixed(2), pct(R[k].offline.recall) + ' / ' + pct(R[k].offline.precision)]),
    M, TOP + 0.1, 7.4, [2.6, 1.05, 1.35, 1.1, 1.3], { size: 13 });
  s.addChart(pres.charts.BAR, [{ name: 'Full-frame AP', labels: names.map(([, m]) => MODEL[m].short),
    values: names.map(([k]) => +D.clips_ap[k].ap.toFixed(2)) }],
  chartOpts({ x: M + 7.8, y: TOP + 0.05, w: CW - 7.8, h: 3.4, barDir: 'bar', barGapWidthPct: 45,
    chartColors: names.map(([, m]) => MODEL[m].color), showValue: true, dataLabelPosition: 'outEnd',
    dataLabelFormatCode: '0.00', title: 'Average precision on full frames', valAxisMinVal: 0, valAxisMaxVal: 1,
    valAxisLabelFormatCode: '0.0', catAxisOrientation: 'maxMin' }));
  card(s, M + 7.8, TOP + 3.65, CW - 7.8, 1.25, { name: 'Crops vs frames card' });
  s.addText([{ text: 'Crops overstate full frames: ', options: { bold: true } },
    { text: 'crop AP 0.80 vs 0.58; full-frame AP 0.50 vs 0.45. Train with more background-only crops next.' }],
  { x: M + 8.0, y: TOP + 3.7, w: CW - 8.2, h: 1.15, fontSize: 14, color: C.text1, valign: 'middle', isTextBox: true,
    margin: 0, objectName: 'Crops vs frames text' });
}

// ================= 6. satellite results =================
divider('Satellite results', 'Synthetic scores overstate real performance, so we built a real test set of 927 aircraft',
  'PlaneNet scores 0.999 AUC on synthetic in-flight validation. The question is how it does on real aircraft in ' +
  'flight from orbit.');
{
  const s = content('The synthetic gap', 'Near-perfect on synthetic samples; far lower on real aircraft in flight',
    'Training: 20 epochs, AdamW, one-cycle schedule. Held-out PlanesNet: 98.6% accuracy (AUC 0.998). Synthetic in-flight ' +
    'validation: 98.8% accuracy (AUC 0.999). data/s2_airborne: 19 aircraft found by a band-offset finder at six ' +
    'airports and labelled by eye, plus 62 clutter and 250 background chips. data/s2_flying: 927 aircraft at 10 ' +
    'airports, built on the next slides.');
  const w = (CW - 0.6) / 3;
  stat(s, M, TOP + 0.3, w, 2.6, '0.999', 'AUC on synthetic in-flight validation (98.8% accuracy)');
  stat(s, M + w + 0.3, TOP + 0.3, w, 2.6, pct(D.s2_airborne['as-is']['recall@0.5 (aircraft+probable)']),
    'of 19 real Sentinel-2 aircraft found at 0.5 (data/s2_airborne)');
  stat(s, M + 2 * (w + 0.3), TOP + 0.3, w, 2.6, pct(D.s2_flying['as-is']['recall@0.5 (aircraft+probable)']),
    'of 927 real Sentinel-2 aircraft found at 0.5 (data/s2_flying, new)');
  bullets(s, ['Sentinel-2 is 10 m per pixel; PlanesNet is about 3 m, so a 6 px window is upsampled to the model\'s 20 px',
    'Both real sets keep only aircraft with a clear three-colour signature, so recall on all aircraft is lower still'],
  M, TOP + 3.3, CW, 1.3, { size: 16 });
}
{
  const s = content('Published labels missed their aircraft', 'Converted to ground positions, they were off by a fixed offset per airport',
    'Flying Airplanes labels (Pamplona Segundo et al. 2021): 1,782 airplanes at 15 European airports, January to June ' +
    '2020, as row/column pixels in mosaics built by their download.py (7x7 blocks of 0.15 x 0.10 degrees). Converting ' +
    'with that geometry and cutting chips from the AWS Sentinel-2 archive put the labels next to, not on, their ' +
    'aircraft: about 200 m east at CDG. Red cross: converted label. Yellow box: the aircraft\'s green copy. The cause ' +
    'could not be found without the authors\' own images.');
  const y = picture(s, img('s2_label_offset.png'), M, TOP + 0.05, CW, 3.529, 'Converted labels beside their aircraft at CDG');
  const steps = [['Search', 'Find blue, green and red copies in a line near each label (150 px)'],
    ['Vote', 'Each image votes for the shift that puts two or more labels on a signature'],
    ['Agree', 'An airport\'s offset is the shift at least 60% of its images agree on'],
    ['Keep', 'Strong signatures at that offset only; duplicates across images removed']];
  const bw = (CW - 3 * 0.3) / 4;
  steps.forEach(([t, d], i) => {
    const x = M + i * (bw + 0.3);
    card(s, x, y + 0.25, bw, 1.35, { name: 'Method step' });
    s.addText([{ text: (i + 1) + '  ' + t, options: { bold: true, fontSize: 15, color: C.accent1, breakLine: true } },
      { text: d, options: { fontSize: 13, color: C.text1 } }], { x: x + 0.18, y: y + 0.32, w: bw - 0.36, h: 1.2,
      valign: 'top', isTextBox: true, margin: 0, objectName: 'Method text' });
  });
}
{
  const s = content('One offset per airport', '10 of 15 airports agree on an offset of 110 m to 1.5 km; 927 aircraft kept',
    'Offsets from the build log (pixels x 10 m). Images agreeing: share of an airport\'s voting images within 15 px of ' +
    'its offset. Dropped: AMS, ARN, ATH, FRA, IST (12-33% agreement within 150 px). With a 400 px search, AMS and FRA ' +
    'give consistent offsets (-4, -158) and (-188, 113) px, but only 61-64% of images agree and about 80% of a checked ' +
    'sample were clear aircraft, so they are left out. Funnel: 1,782 labels; 1,189 at the 10 kept airports; 927 ' +
    'matched to a strong signature (score >= 15) after removing duplicates.');
  const by = D.s2_build['by airport'];
  const kept = Object.entries(by).filter(([, v]) => v['offset px']).map(([k, v]) => [k, Math.hypot(...v['offset px']) * 10, v])
    .sort((a, b) => a[1] - b[1]);
  s.addChart(pres.charts.BAR, [{ name: 'Offset (m)', labels: kept.map(([k, , v]) => k + ' (' + v['files agreeing'].split('%')[0] + '%)'), values: kept.map(([, m]) => Math.round(m)) }],
  chartOpts({ x: M, y: TOP + 0.05, w: 7.0, h: 4.85, barDir: 'bar', barGapWidthPct: 40, chartColors: [T.accent1],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '#,##0" m"', title: 'Label offset by airport (images agreeing)',
    valAxisMinVal: 0, valAxisMaxVal: 1800, valAxisLabelFormatCode: '#,##0', catAxisOrientation: 'maxMin' }));
  const total = Object.values(by).reduce((a, v) => a + v.labels, 0);
  const keptLabels = kept.reduce((a, [, , v]) => a + v.labels, 0);
  s.addChart(pres.charts.BAR, [{ name: 'Labels', labels: ['Labels, 15 airports', 'At 10 airports with an offset',
    'Matched to a strong signature'], values: [total, keptLabels, D.s2_build.aircraft] }],
  chartOpts({ x: M + 7.4, y: TOP + 0.05, w: CW - 7.4, h: 3.2, barDir: 'bar', barGapWidthPct: 35,
    chartColors: ['86B6EF', '5598E7', T.accent1], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '#,##0',
    title: 'From labels to test aircraft', valAxisMinVal: 0, valAxisMaxVal: 2200, valAxisLabelFormatCode: '#,##0',
    catAxisOrientation: 'maxMin', catAxisLabelFontSize: 12 }));
  stat(s, M + 7.4, TOP + 3.45, CW - 7.4, 1.45, '72 of 80', 'random chips of the final set show a clear aircraft (by eye)',
    { bigSize: 30 });
}
{
  const s = content('The 927-aircraft test set', 'Aircraft over fields, towns and sea at 10 airports, plus 2,134 background chips',
    'Each chip is 64x64 px at 10 m, centred on the aircraft\'s green copy, 8-bit reflectance / 0.3, the format of ' +
    'data/s2_airborne. Chips are not committed; python -m satellite.build_s2_flying data/s2_flying --chips-from-labels ' +
    're-cuts them from labels.csv (checked byte-identical on 279 chips). Contains modified Copernicus Sentinel data (2020).');
  const y = picture(s, img('s2_flying_examples.png'), M + 0.9, TOP - 0.05, CW - 1.8, 2.727, 'Aircraft chips at ten airports');
  caption(s, 'Background chips: random points at least 1 km from any label', M + 0.9, y + 0.05, CW - 1.8, 0.3,
    { color: HEX.muted });
  picture(s, img('s2_flying_background.png'), M + 0.9, y + 0.35, (CW - 1.8) * 0.62, 5.0, 'Background chips');
}
{
  const s = content('Satellite results on 927 aircraft', '71% found at 0.7% false alarms; realigning the bands no longer helps',
    'As-is: the 60 m (6 px) square around the green copy, upsampled to 20 px. Realigned: red and blue shifted onto the ' +
    'green copy first, using the recorded copy positions. s2_airborne: AUC 0.99 vs background, 0.88 vs clutter. ' +
    's2_flying: AUC 0.96 vs background (as-is), 0.94 realigned; false alarms 0.7% on 2,134 background chips. Recall by ' +
    'airport ranges from 53% (DUB, 66 aircraft) to 77% (BRU, DUS). The 63% realigned on 19 aircraft was within noise.');
  const sets = [['s2_airborne', '19 aircraft (s2_airborne)'], ['s2_flying', '927 aircraft (s2_flying)']];
  s.addChart(pres.charts.BAR, ['as-is', 'realigned'].map((k) => ({ name: k === 'as-is' ? 'As-is' : 'Bands realigned',
    labels: sets.map((x) => x[1]), values: sets.map(([d]) => D[d][k]['recall@0.5 (aircraft+probable)']) })),
  chartOpts({ x: M, y: TOP + 0.05, w: 5.4, h: 4.85, barDir: 'col', barGapWidthPct: 60, chartColors: [T.accent1, '86B6EF'],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0%', showLegend: true, legendPos: 'b',
    title: 'Aircraft found at 0.5', valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0%', catAxisLabelFontSize: 13 }));
  const ra = D.s2_flying['as-is']['recall@0.5 by airport'];
  const rows = Object.entries(ra).map(([k, v]) => { const [a, n] = v.split('/').map(Number); return [k, a / n, n]; })
    .sort((a, b) => b[1] - a[1]);
  s.addChart(pres.charts.BAR, [{ name: 'Recall', labels: rows.map(([k, , n]) => k + ' (' + n + ')'), values: rows.map((r) => r[1]) }],
    chartOpts({ x: M + 5.8, y: TOP + 0.05, w: CW - 5.8, h: 4.85, barDir: 'bar', barGapWidthPct: 40, chartColors: [T.accent1],
      showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0%', title: 'Found at 0.5, by airport (aircraft)',
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0%', catAxisOrientation: 'maxMin' }));
}

{
  const cv = D.s2net['cross-validation by airport'];
  const s = content('S2Net: a model for Sentinel-2', 'Cross-validated by airport it finds ' +
    pct(cv['recall@0.5 (aircraft+probable)']) + ' at ' + (100 * cv['false positive rate@0.5, background']).toFixed(1) +
    '% false alarms (PlaneNet: 71%, 0.7%)',
    'satellite/s2_model.py and satellite/train_s2.py. 5 folds by airport: every chip is scored by a model that never ' +
    'saw its airport; folds ' + cv.folds.map((f) => f.join('+')).join(', ') + '. Partly circular: data/s2_flying keeps ' +
    'only aircraft whose signature a band-offset finder saw clearly, which is what S2Net learns. Every airport ' +
    'scores 97% or more. Training: 30 epochs, AdamW, one-cycle schedule, positives weighted to balance classes, CPU.');
  bullets(s, [
    [{ text: 'Window: ', options: { bold: true } }, { text: '48×48 px (480 m) holds all three copies; PlaneNet sees 60 m' }],
    [{ text: 'Network: ', options: { bold: true } }, { text: 'three blocks of 3×3 convolutions with batch norm, 290k parameters' }],
    [{ text: 'Positives: ', options: { bold: true } }, { text: '927 real aircraft of data/s2_flying' }],
    [{ text: 'Negatives: ', options: { bold: true } }, { text: 'background chips as cut, and re-centred on their strongest colour spot' }],
    [{ text: 'Augmentation: ', options: { bold: true } }, { text: 'rotations, flips, 2 px shifts, brightness and contrast' }],
  ], M, TOP + 0.2, 6.6, 4.4, { size: 16, space: 12 });
  const x = M + 7.0;
  const w = CW - 7.0;
  stat(s, x, TOP + 0.1, w, 1.5, pct(cv['recall@0.5 (aircraft+probable)']), 'of 927 aircraft found at 0.5, each scored ' +
    'by a model that never saw its airport', { bigSize: 36 });
  stat(s, x, TOP + 1.8, w, 1.5, (100 * cv['false positive rate@0.5, background']).toFixed(1) + '%', 'false alarms on 2,134 ' +
    'background chips (' + (100 * cv['false positive rate@0.5, background (peak-centred)']).toFixed(1) + '% when centred on a colourful spot)',
  { bigSize: 36 });
  stat(s, x, TOP + 3.5, w, 1.4, cv['auc vs background'].toFixed(3), 'AUC against background (PlaneNet: 0.96)', { bigSize: 36 });
}
{
  const t = D.s2net['test data/s2_airborne'];
  const m = D.matched_fpr;
  const s = content('Independent test: S2Net finds 18 of 19', 'On 2025 scenes from other airports it beats PlaneNet at every false-alarm level',
    'data/s2_airborne: 2025, six airports (LHR, LGW, CDG, FRA, ORD, JFK), candidates from a different band-offset finder, ' +
    'labelled by eye: 14 aircraft, 5 probable, 62 clutter (cars, roofs, lights), 250 background. S2Net trained on all ' +
    'of data/s2_flying. Matched rates: for each model the threshold is the highest that keeps clutter false alarms at ' +
    'or below 5%, 8.1% (PlaneNet\'s rate at 0.5) and 15%. At 0.5 S2Net raises slightly more false alarms (clutter 11%, ' +
    'background 1.2%) than PlaneNet (8%, 0.8%). With 19 aircraft every figure has wide error bars, and both test sets ' +
    'lean towards clear signatures.');
  const cats = ['Clutter false alarms ≤ 5%', '≤ 8.1%', '≤ 15%'];
  s.addChart(pres.charts.BAR, ['S2Net', 'PlaneNet'].map((n) => ({ name: n, labels: cats,
    values: ['0.05', '0.081', '0.15'].map((f) => m[n][f]) })),
  chartOpts({ x: M, y: TOP + 0.05, w: 7.4, h: 4.85, barDir: 'col', barGapWidthPct: 60, chartColors: [T.accent2, T.accent1],
    showValue: true, dataLabelPosition: 'outEnd', showLegend: true, legendPos: 'b', title: 'Aircraft found (of 19)',
    valAxisMinVal: 0, valAxisMaxVal: 20, valAxisMajorUnit: 5, catAxisLabelFontSize: 13 }));
  const pa = D.s2_airborne['as-is'];
  table(s, ['', 'S2Net', 'PlaneNet'], [
    ['Found at 0.5', pct(t['recall@0.5 (aircraft+probable)']), pct(pa['recall@0.5 (aircraft+probable)'])],
    ['AUC vs clutter', t['auc vs clutter'].toFixed(2), pa['auc vs clutter'].toFixed(2)],
    ['AUC vs background', t['auc vs background'].toFixed(3), pa['auc vs background'].toFixed(3)],
    ['False alarms at 0.5, clutter', pct(t['false positive rate@0.5, clutter']), pct(pa['false positive rate@0.5, clutter'])],
    ['False alarms at 0.5, background', (100 * t['false positive rate@0.5, background']).toFixed(1) + '%',
      (100 * pa['false positive rate@0.5, background']).toFixed(1) + '%'],
  ], M + 7.8, TOP + 0.1, CW - 7.8, [2.35, 1.0, CW - 7.8 - 3.35], { size: 13 });
}

// ================= 7. repository and next steps =================
divider('Repository and next steps', 'What was fixed, how to rerun everything, what is still open',
  'PR rohanshenoy/planesnet-detector#1 holds the original work; branch claude/validation-sets adds the test sets, ' +
  'the 50-epoch run and the fixes below.');
{
  const s = content('Problems found and fixed', 'Three .gitignore rules had silently dropped files from the repo',
    'Each fix is a commit on claude/fervent-pascal-x2ge7z (PR #1) or claude/validation-sets, with tests where code ' +
    'changed. The P2 config was rebuilt from the config stored inside the trained checkpoint: 2,740,032 parameters, ' +
    'identical 593 tensor names and shapes, 297/593 tensors transfer from yolo11n.pt.');
  table(s, ['Problem', 'Effect', 'Fix'], [
    ['.gitignore models/ matched any folder named models', 'sky/models/yolo11-p2.yaml never committed',
      'Rule anchored to /models/; config rebuilt from the checkpoint'],
    ['.gitignore *_detections.png matched a figure', 'Report figure missing', 'Restored; report_figs/ re-included'],
    ['.gitignore *.pt glued to the previous line', 'Weights outside models/ not ignored', 'Rule on its own line'],
    ['AOT parts share flights', 'Training on part 3 would leak into the test set', '--exclude-flights; part 2 as test set'],
    ['Network resets during downloads', 'About a quarter of crops skipped', 'Retries with backoff (part 3: 0 failures)'],
    ['Part 2 labels spell a bird "BIrd"', '61 bird boxes dropped instead of ignored', 'Mapped to bird'],
    ['Ultralytics 8.4 nests relative run folders', 'Runs saved under runs/detect/runs/sky/', 'Absolute project path'],
    ['evaluate_clips still called the old fetch()', 'Every clip download would crash', 'Part passed through; regression test'],
  ], M, TOP + 0.1, CW, [4.2, 3.9, 4.03], { size: 13 });
}
{
  const s = content('Reproducing the work', '34 tests run in seconds on synthetic data; every number has a command',
    'Tests: python -m pytest -q (34 passed, about 15 s, no downloads). Local environment: .venv with requirements.txt ' +
    'plus rasterio. Trained weights are git-ignored in models/ (see models/README.md); downloaded data lives under ' +
    'data/ and is git-ignored except data/s2_airborne and data/s2_flying/labels.csv.');
  const cmds = [
    'python -m pytest -q',
    'python -m sky.prepare_aot part2.csv data/aot_part2 --part part2 --train 0 --val 1500 --val-fraction 1',
    'python -m sky.prepare_aot part3.csv data/aot_train_p3 --part part3 --train 9000 --val 0 --exclude-flights part2.csv',
    'python -m sky.finetune data/aot_train/temporal.yaml --temporal --weights yolo11s.pt --epochs 50 --device mps',
    'python -m sky.evaluate models/sky_yolo11s_3frame_e50.pt data/aot_part2/temporal --meta data/aot_part2/meta.json',
    'python -m satellite.build_s2_flying data/s2_flying --chips-from-labels',
    'python -m satellite.evaluate_s2 models/satellite_planenet.pt data/s2_flying',
    'python -m satellite.train_s2 data/s2_flying --out models/s2net.pt --test data/s2_airborne',
    'python -m sky.evaluate_clips part1.csv data/aot_clips --clips 8 --model s=models/sky_yolo11s_3frame_e50.pt:0.226:2',
  ];
  card(s, M, TOP + 0.1, CW, 4.0, { name: 'Command card' });
  s.addText(cmds.map((c, i) => ({ text: c, options: { breakLine: i < cmds.length - 1, paraSpaceAfter: 7 } })),
    { x: M + 0.3, y: TOP + 0.3, w: CW - 0.6, h: 3.6, fontFace: 'Courier New', fontSize: 12, color: C.text1, valign: 'top',
      isTextBox: true, margin: 0, objectName: 'Commands' });
  caption(s, 'Full write-up: docs/report.md · agent guide: CLAUDE.md · dataset notes: data/s2_flying/README.md',
    M, TOP + 4.35, CW, 0.4, { color: HEX.muted });
}
{
  const s = content('Limitations', 'The numbers are real, but each has a known bias or gap',
    'Each item here is also in docs/report.md under Limitations.');
  const items = [
    ['Crops overstate full frames', 'Sky AP 0.80 on crops but 0.50 on full frames; 8 clips give wide error bars.'],
    ['Epochs not isolated', 'The part-1-only run separates part 3 data, not epochs (needs 50 epochs on 1,548 crops).'],
    ['Satellite sets lean on band-offset finders', 'Both keep clear signatures; S2Net cross-validation is partly circular.'],
    ['Small independent satellite test', '19 aircraft in data/s2_airborne; labels of s2_flying about 90% clean.'],
    ['One training run per model', 'Differences of a few points in a subgroup are within run-to-run noise.'],
  ];
  items.forEach(([t, d], i) => {
    const y = TOP + 0.1 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.1, w: 0.55, h: 0.55, fill: { color: C.accent4 }, line: { type: 'none' },
      objectName: 'Limitation badge' });
    s.addText('!', { x: M, y: y + 0.1, w: 0.55, h: 0.55, align: 'center', valign: 'middle', fontSize: 20, bold: true,
      color: C.text1, isTextBox: true, margin: 0, objectName: 'Limitation mark' });
    s.addText([{ text: t, options: { bold: true, fontSize: 17, breakLine: true } }, { text: d, options: { fontSize: 15 } }],
      { x: M + 0.8, y, w: CW - 0.8, h: 0.85, valign: 'middle', color: C.text1, isTextBox: true, margin: 0,
        objectName: 'Limitation text' });
  });
}
{
  const s = pres.addSlide({ masterName: 'CLOSING', sectionTitle: 'Repository and next steps' });
  s.addText('Next steps', { placeholder: 'title' });
  s.addText('In order of expected value', { placeholder: 'body', bullet: false });
  const steps = [
    ['Train the sky model for full frames', 'Add background-only and hard-negative crops mined from full frames; rescore clips'],
    ['Isolate epochs', 'Train 50 epochs on 1,548 part 1 crops and compare with the 12-epoch run'],
    ['Test S2Net without a finder', 'Sentinel-2 aircraft located from ADS-B tracks, so faint ones are included'],
    ['Harder S2Net negatives', 'Use finder false positives (cars, roofs, lights) as training negatives'],
    ['Merge', 'Merge PR #1 and the validation-sets PR; keep the 93 MB PlanesNet zip out of master'],
  ];
  steps.forEach(([t, d], i) => {
    const y = TOP + 0.1 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.1, w: 0.6, h: 0.6, fill: { color: C.accent2 }, line: { type: 'none' },
      objectName: 'Step badge' });
    s.addText(String(i + 1), { x: M, y: y + 0.1, w: 0.6, h: 0.6, align: 'center', valign: 'middle', fontSize: 20, bold: true,
      color: C.background1, isTextBox: true, margin: 0, objectName: 'Step number' });
    s.addText([{ text: t, options: { bold: true, fontSize: 18, color: C.background1, breakLine: true } },
      { text: d, options: { fontSize: 15, color: C.background2 } }], { x: M + 0.85, y, w: CW - 0.85, h: 0.85,
      valign: 'middle', isTextBox: true, margin: 0, objectName: 'Step text' });
  });
  s.addNotes('Done since the first version of this deck: the 50-epoch model on full frames, a part-1-only 50-epoch run, ' +
    'S2Net, and the wider search for dropped airports. The PlanesNet data branch holds a 93 MB zip that should not be merged.');
}

// ================= write =================
(async () => {
  await pres.writeFile({ fileName: B.OUT });
  await B.applyTheme(B.OUT, B.THEME);
  console.log('wrote', B.OUT);
})();
