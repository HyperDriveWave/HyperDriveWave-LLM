/*
 * 趋势图内联 SVG 渲染 —— 移植自 SmartGasTurbine 的 llm_inference.html。
 *
 * 为什么是内联 SVG 而不是图片：
 *   · 矢量图自适应容器宽度，不糊
 *   · 不用把 base64 图片塞进 JSON（一张 PNG 53KB，会让响应体大一个量级）
 *   · 可以带悬停提示
 *
 * 图上画三样东西，语义与 SmartGasTurbine 一致：
 *   实线  实际采样        #1769aa
 *   虚线  最小二乘拟合     #f59e0b  ← 仅视觉参考，模型引用数值时必须用真实采样值
 *   标记  极值圆点 + 斜率变号菱形，变号点带竖直虚线引导线
 *
 * 安全：所有文本经 escapeSvgText 转义后才拼进 SVG 字符串。
 * 数据来自后端计算的数值，但「测点描述」「detail」是中文自由文本，必须转义。
 */
(function (global) {
  "use strict";

  var WIDTH = 820;
  var HEIGHT = 260;
  var PAD = { left: 62, right: 20, top: 18, bottom: 44 };

  function escapeSvgText(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(value) {
    var n = Number(value);
    if (!isFinite(n)) return "--";
    var abs = Math.abs(n);
    if (abs === 0) return "0";
    if (abs >= 1000 || abs < 0.01) return n.toExponential(3);
    return String(Number(n.toPrecision(6)));
  }

  // 时间统一按 UTC+8 显示（后端也是这个口径，图上不能出现两套时间）
  function localTime(isoText) {
    var moment = new Date(isoText);
    if (isNaN(moment.getTime())) return String(isoText || "");
    var shifted = new Date(moment.getTime() + 8 * 3600 * 1000);
    function pad(n) { return String(n).padStart(2, "0"); }
    return (
      pad(shifted.getUTCMonth() + 1) + "-" + pad(shifted.getUTCDate()) +
      " " + pad(shifted.getUTCHours()) + ":" + pad(shifted.getUTCMinutes())
    );
  }

  function normalize(points) {
    if (!Array.isArray(points)) return [];
    var out = [];
    for (var i = 0; i < points.length; i += 1) {
      var p = points[i] || {};
      var t = new Date(p.time).getTime();
      var v = Number(p.value);
      if (isNaN(t) || !isFinite(v)) continue;
      out.push({ timeMs: t, value: v, time: p.time, type: p.type, detail: p.detail });
    }
    out.sort(function (a, b) { return a.timeMs - b.timeMs; });
    return out;
  }

  // 变号点用菱形 + 引导线，极值点用圆点——形状区分比颜色区分更可靠
  var MARK_COLORS = {
    global_max: "#067647",
    global_min: "#b42318",
    latest: "#7c3aed",
    slope_sign_change: "#b54708",
  };

  function render(chart) {
    var actual = normalize(chart.series);
    var fit = normalize(chart.fit_series);
    var specials = normalize(chart.special_points || chart.specialPoints);
    if (!actual.length && !fit.length) {
      return '<div class="trend-chart-empty">暂无可绘制的趋势数据</div>';
    }

    var all = actual.concat(fit);
    var minX = all[0].timeMs, maxX = all[all.length - 1].timeMs;
    var minY = all[0].value, maxY = all[0].value;
    for (var i = 1; i < all.length; i += 1) {
      if (all[i].timeMs < minX) minX = all[i].timeMs;
      if (all[i].timeMs > maxX) maxX = all[i].timeMs;
      if (all[i].value < minY) minY = all[i].value;
      if (all[i].value > maxY) maxY = all[i].value;
    }
    if (Math.abs(maxY - minY) < 1e-12) {
      // 完全平直：给一点纵向余量，否则线会贴在边上
      minY -= 1; maxY += 1;
    } else {
      var margin = (maxY - minY) * 0.08;
      minY -= margin; maxY += margin;
    }

    var plotW = WIDTH - PAD.left - PAD.right;
    var plotH = HEIGHT - PAD.top - PAD.bottom;
    var spanX = Math.max(1, maxX - minX);
    var spanY = Math.max(1e-12, maxY - minY);
    function xOf(p) { return PAD.left + ((p.timeMs - minX) / spanX) * plotW; }
    function yOf(p) { return PAD.top + (1 - (p.value - minY) / spanY) * plotH; }
    function pathOf(list) {
      return list.map(function (p, i) {
        return (i ? "L" : "M") + xOf(p).toFixed(1) + "," + yOf(p).toFixed(1);
      }).join(" ");
    }

    var unit = (chart.point && chart.point.unit) || "";
    var parts = [];

    // 背景与横向网格
    parts.push('<rect x="0" y="0" width="' + WIDTH + '" height="' + HEIGHT + '" fill="#ffffff"/>');
    var ticks = 4;
    for (var t = 0; t <= ticks; t += 1) {
      var ratio = t / ticks;
      var gy = PAD.top + ratio * plotH;
      var gv = maxY - ratio * (maxY - minY);
      parts.push('<line x1="' + PAD.left + '" y1="' + gy.toFixed(1) +
                 '" x2="' + (WIDTH - PAD.right) + '" y2="' + gy.toFixed(1) +
                 '" stroke="#eef1f5"/>');
      parts.push('<text x="' + (PAD.left - 8) + '" y="' + (gy + 4).toFixed(1) +
                 '" text-anchor="end" font-size="11" fill="#667085">' +
                 escapeSvgText(formatNumber(gv)) + '</text>');
    }

    // 坐标轴
    parts.push('<line x1="' + PAD.left + '" y1="' + (HEIGHT - PAD.bottom) +
               '" x2="' + (WIDTH - PAD.right) + '" y2="' + (HEIGHT - PAD.bottom) +
               '" stroke="#c9d3df"/>');
    parts.push('<line x1="' + PAD.left + '" y1="' + PAD.top +
               '" x2="' + PAD.left + '" y2="' + (HEIGHT - PAD.bottom) +
               '" stroke="#c9d3df"/>');

    // 时间轴：只标首尾，中间靠悬停看
    parts.push('<text x="' + PAD.left + '" y="' + (HEIGHT - 14) +
               '" text-anchor="start" font-size="11" fill="#667085">' +
               escapeSvgText(localTime(new Date(minX).toISOString())) + '</text>');
    parts.push('<text x="' + (WIDTH - PAD.right) + '" y="' + (HEIGHT - 14) +
               '" text-anchor="end" font-size="11" fill="#667085">' +
               escapeSvgText(localTime(new Date(maxX).toISOString())) + '</text>');

    // 拟合虚线画在实线**下面**：它是参考，不该盖住真实数据
    if (fit.length >= 2) {
      parts.push('<path d="' + pathOf(fit) + '" fill="none" stroke="#f59e0b" stroke-width="2"' +
                 ' stroke-dasharray="6 4" stroke-linejoin="round" stroke-linecap="round"/>');
    }
    if (actual.length >= 2) {
      parts.push('<path d="' + pathOf(actual) + '" fill="none" stroke="#1769aa" stroke-width="2.2"' +
                 ' stroke-linejoin="round" stroke-linecap="round"/>');
    }

    // 特殊点
    specials.forEach(function (p) {
      var x = xOf(p), y = yOf(p);
      var color = MARK_COLORS[p.type] || "#b54708";
      var label = escapeSvgText(p.type || "");
      var tip = escapeSvgText(
        (p.type || "") + " " + localTime(p.time) + " " + formatNumber(p.value) + unit +
        (p.detail ? "，" + p.detail : "")
      );
      if (p.type === "slope_sign_change") {
        // 引导线让「这个点发生在什么时候」一眼可读
        parts.push('<line x1="' + x.toFixed(1) + '" y1="' + PAD.top +
                   '" x2="' + x.toFixed(1) + '" y2="' + (HEIGHT - PAD.bottom) +
                   '" stroke="' + color + '" stroke-width="1" stroke-dasharray="4 5" opacity="0.45"/>');
        parts.push('<rect x="' + (x - 4).toFixed(1) + '" y="' + (y - 4).toFixed(1) +
                   '" width="8" height="8" transform="rotate(45 ' + x.toFixed(1) + ' ' + y.toFixed(1) +
                   ')" fill="' + color + '" stroke="#fff" stroke-width="1.2"><title>' + tip + '</title></rect>');
      } else {
        parts.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) +
                   '" r="4.2" fill="' + color + '" stroke="#fff" stroke-width="1.5"><title>' + tip +
                   '</title></circle>');
      }
    });

    parts.push('<text x="' + PAD.left + '" y="12" font-size="11" fill="#667085">' +
               escapeSvgText(unit) + '</text>');

    return '<svg class="trend-chart" viewBox="0 0 ' + WIDTH + ' ' + HEIGHT +
           '" preserveAspectRatio="xMidYMid meet" role="img">' + parts.join("") + '</svg>';
  }

  global.HDWTrendChart = { render: render, escapeSvgText: escapeSvgText };
})(window);
