(function () {
  'use strict';

  var App = window.App = window.App || {};

  App.state = {
    health: null,
    datasets: [],
    runs: [],
    currentDataset: null,
    currentRun: null,
    activeJobId: null,
    eventSource: null,
    lastTerminalJob: null,
    preview: null,
    previewTimer: null,
    compareSelected: {}
  };

  App.VIEW_META = {
    overview: { title: '概览', eyebrow: 'WORKSPACE', subtitle: '数据集、训练与验证的统一入口' },
    datasets: { title: '数据集', eyebrow: 'DATA', subtitle: '拖入、检查、预览并准备 Avenue 结构数据' },
    train: { title: '训练', eyebrow: 'TRAIN', subtitle: '配置实验并实时观察 loss 与验证指标' },
    validate: { title: '验证', eyebrow: 'EVALUATE', subtitle: '逐帧指标、热力图、解释与复评' },
    compare: { title: '对比', eyebrow: 'COMPARE', subtitle: '横向比较不同 prompt / fusion / 运行' }
  };

  App.$ = function (selector, root) {
    return (root || document).querySelector(selector);
  };

  App.$$ = function (selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  };

  App.escape = function (value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  };

  App.encode = function (value) {
    return encodeURIComponent(String(value == null ? '' : value));
  };

  App.fmt = function (value, digits) {
    if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) {
      return '未定义';
    }
    return Number(value).toFixed(digits === undefined ? 4 : digits);
  };

  App.fmtInt = function (value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '-';
    }
    return String(Math.round(Number(value)));
  };

  App.fmtDuration = function (seconds) {
    var total = Math.max(0, Math.round(Number(seconds) || 0));
    var minutes = Math.floor(total / 60);
    var rest = total % 60;
    if (minutes > 0) {
      return minutes + 'm ' + rest + 's';
    }
    return rest + 's';
  };

  App.fmtDate = function (timestamp) {
    if (!timestamp) {
      return '-';
    }
    var date = new Date(Number(timestamp) * 1000);
    return date.toLocaleString('zh-CN', { hour12: false });
  };

  App.ago = function (timestamp) {
    if (!timestamp) {
      return '-';
    }
    var seconds = Math.max(0, Date.now() / 1000 - Number(timestamp));
    if (seconds < 60) { return '刚刚'; }
    if (seconds < 3600) { return Math.floor(seconds / 60) + ' 分钟前'; }
    if (seconds < 86400) { return Math.floor(seconds / 3600) + ' 小时前'; }
    return Math.floor(seconds / 86400) + ' 天前';
  };

  App.api = async function (path, options) {
    var opts = options || {};
    var headers = opts.headers || {};
    if (opts.body && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
    var response = await fetch(path, Object.assign({}, opts, { headers: headers }));
    var text = await response.text();
    var data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        data = { error: text };
      }
    }
    if (!response.ok) {
      var message = data && data.error ? data.error : ('HTTP ' + response.status);
      throw new Error(message);
    }
    return data || {};
  };

  App.toast = function (message, kind) {
    var host = App.$('#toasts');
    if (!host) {
      return;
    }
    var node = document.createElement('div');
    node.className = 'toast ' + (kind || '');
    node.textContent = message;
    host.appendChild(node);
    setTimeout(function () {
      node.style.opacity = '0';
      node.style.transform = 'translateY(8px)';
      node.style.transition = '0.25s ease';
      setTimeout(function () { node.remove(); }, 260);
    }, kind === 'error' ? 6000 : 3600);
  };

  App.statusLabel = function (status) {
    var labels = {
      queued: '排队中',
      running: '运行中',
      stopping: '停止中',
      stopped: '已停止',
      completed: '已完成',
      failed: '失败'
    };
    return labels[status] || status || '未知';
  };

  App.statusBadge = function (status) {
    var cls = 'badge';
    if (status === 'running' || status === 'queued') { cls += ' info'; }
    if (status === 'completed') { cls += ' ok'; }
    if (status === 'failed') { cls += ' danger'; }
    if (status === 'stopped' || status === 'stopping') { cls += ' warn'; }
    return '<span class="' + cls + '">' + App.escape(App.statusLabel(status)) + '</span>';
  };

  App.kindLabel = function (kind) {
    var labels = { train: '训练', eval: '复评', zero_shot: 'Zero-shot', prepare: '数据准备', demo: 'Demo' };
    return labels[kind] || kind || '-';
  };

  App.aucClass = function (value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '';
    }
    if (Number(value) >= 0.65) { return 'auc-good'; }
    if (Number(value) >= 0.50) { return 'auc-mid'; }
    return 'auc-bad';
  };

  App.metricCard = function (label, value, sub, cls) {
    return '<div class="stat">'
      + '<div class="stat-label">' + App.escape(label) + '</div>'
      + '<div class="stat-value ' + (cls || '') + '">' + value + '</div>'
      + (sub ? '<div class="stat-sub">' + App.escape(sub) + '</div>' : '')
      + '</div>';
  };

  App.empty = function (icon, text) {
    return '<div class="empty"><div class="empty-icon">' + icon + '</div><p>' + App.escape(text) + '</p></div>';
  };

  App.table = function (headers, rows) {
    if (!rows || !rows.length) {
      return App.empty('·', '暂无数据');
    }
    var html = '<table><thead><tr>';
    headers.forEach(function (header) {
      html += '<th>' + header + '</th>';
    });
    html += '</tr></thead><tbody>';
    rows.forEach(function (row) {
      html += '<tr>';
      row.forEach(function (cell) {
        html += '<td>' + cell + '</td>';
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
  };

  App.showView = function (name) {
    var meta = App.VIEW_META[name] || App.VIEW_META.overview;
    App.$$('.view').forEach(function (view) { view.classList.add('hidden'); });
    var target = document.getElementById('view-' + name);
    if (target) {
      target.classList.remove('hidden');
    }
    App.$$('.nav-item').forEach(function (item) {
      item.classList.toggle('active', item.getAttribute('data-view') === name);
    });
    var title = App.$('#view-title');
    var eyebrow = App.$('#view-eyebrow');
    var subtitle = App.$('#view-subtitle');
    if (title) { title.textContent = meta.title; }
    if (eyebrow) { eyebrow.textContent = meta.eyebrow; }
    if (subtitle) { subtitle.textContent = meta.subtitle; }
    if (location.hash !== '#' + name) {
      history.replaceState(null, '', '#' + name);
    }
    App.state.currentView = name;

    if (name === 'overview' && App.loadOverview) { App.loadOverview(); }
    if (name === 'datasets' && App.loadDatasets) { App.loadDatasets(); }
    if (name === 'train' && App.loadDatasets) {
      App.loadDatasets().then(function () {
        if (App.populateTrainDatasets) { App.populateTrainDatasets(); }
      });
    }
    if (name === 'validate') {
      if (App.loadRuns) { App.loadRuns(); }
      if (App.loadJobs) { App.loadJobs(); }
    }
    if (name === 'compare' && App.loadCompare) { App.loadCompare(); }
  };

  App.setActiveJob = function (job) {
    var chip = App.$('#active-job-chip');
    if (!chip) { return; }
    if (job && (job.status === 'running' || job.status === 'queued' || job.status === 'stopping')) {
      chip.classList.remove('hidden');
      chip.textContent = App.kindLabel(job.kind) + ' · ' + (job.run_name || job.id);
    } else {
      chip.classList.add('hidden');
      chip.textContent = '';
    }
  };

  App.openModal = function (html) {
    var modal = App.$('#modal');
    var body = App.$('#modal-body');
    if (!modal || !body) { return; }
    body.innerHTML = html;
    modal.classList.remove('hidden');
  };

  App.closeModal = function () {
    var modal = App.$('#modal');
    if (modal) { modal.classList.add('hidden'); }
  };

  App.zoom = function (src, caption) {
    App.openModal('<figure style="margin:0">'
      + '<img src="' + App.escape(src) + '" alt="' + App.escape(caption || '') + '">'
      + '<figcaption class="muted" style="padding-top:8px;font-family:var(--mono);font-size:11px">'
      + App.escape(caption || '') + '</figcaption></figure>');
  };

  App.drawTrainingChart = function (history) {
    var canvas = App.$('#train-chart');
    if (!canvas) { return; }
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#070c17';
    ctx.fillRect(0, 0, W, H);

    var rows = (history || []).filter(function (item) {
      return item && (item.train || item.val);
    });
    if (!rows.length) {
      ctx.fillStyle = '#65779e';
      ctx.font = '13px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'center';
      ctx.fillText('等待第一个 epoch 的指标…', W / 2, H / 2);
      return;
    }

    var pad = { l: 54, r: 56, t: 34, b: 30 };
    var plotW = W - pad.l - pad.r;
    var plotH = H - pad.t - pad.b;
    var epochs = rows.map(function (item) { return Number(item.epoch) || 0; });
    var minE = Math.min.apply(null, epochs);
    var maxE = Math.max.apply(null, epochs);
    var xOf = function (epoch) {
      if (maxE === minE) { return pad.l + plotW / 2; }
      return pad.l + ((epoch - minE) / (maxE - minE)) * plotW;
    };

    var leftValues = [];
    rows.forEach(function (item) {
      if (item.train) {
        leftValues.push(Number(item.train.loss) || 0);
        leftValues.push(Number(item.train.bce) || 0);
        leftValues.push(Number(item.train.contrastive) || 0);
      }
    });
    var leftMax = Math.max(0.1, Math.max.apply(null, leftValues.length ? leftValues : [1]));
    leftMax = leftMax * 1.15;
    var yLeft = function (value) { return pad.t + plotH - (Number(value) / leftMax) * plotH; };
    var yRight = function (value) { return pad.t + plotH - Math.max(0, Math.min(1, Number(value))) * plotH; };

    ctx.strokeStyle = 'rgba(35,50,80,0.7)';
    ctx.lineWidth = 1;
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'right';
    for (var g = 0; g <= 4; g += 1) {
      var y = pad.t + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
      ctx.fillStyle = '#65779e';
      ctx.fillText((leftMax * (1 - g / 4)).toFixed(2), pad.l - 7, y + 3);
      ctx.textAlign = 'left';
      ctx.fillText((1 - g / 4).toFixed(2), pad.l + plotW + 7, y + 3);
      ctx.textAlign = 'right';
    }

    ctx.fillStyle = '#91a2c7';
    ctx.textAlign = 'left';
    ctx.fillText('epoch', pad.l, H - 9);
    ctx.textAlign = 'right';
    ctx.fillText('loss', pad.l - 7, 18);
    ctx.textAlign = 'left';
    ctx.fillText('AUC', pad.l + plotW + 7, 18);

    var drawSeries = function (getter, color, yFn, width) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width || 2;
      var started = false;
      rows.forEach(function (item) {
        var value = getter(item);
        if (value === null || value === undefined || Number.isNaN(Number(value))) { return; }
        var px = xOf(Number(item.epoch) || 0);
        var py = yFn(value);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      });
      if (started) { ctx.stroke(); }
    };

    drawSeries(function (item) { return item.train && item.train.loss; }, '#5eead4', yLeft, 2.4);
    drawSeries(function (item) { return item.train && item.train.bce; }, '#fbbf24', yLeft, 1.6);
    drawSeries(function (item) { return item.train && item.train.contrastive; }, '#818cf8', yLeft, 1.6);
    drawSeries(function (item) { return item.val && item.val.frame_auc; }, '#60a5fa', yRight, 2.2);
    drawSeries(function (item) { return item.val && item.val.clip_auc; }, '#34d399', yRight, 2.2);

    var legend = [
      ['loss', '#5eead4'], ['bce', '#fbbf24'], ['contrastive', '#818cf8'],
      ['val frame AUC', '#60a5fa'], ['val clip AUC', '#34d399']
    ];
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    var lx = pad.l;
    legend.forEach(function (entry) {
      ctx.fillStyle = entry[1];
      ctx.fillRect(lx, 10, 8, 8);
      ctx.fillStyle = '#91a2c7';
      ctx.fillText(entry[0], lx + 12, 18);
      lx += ctx.measureText(entry[0]).width + 34;
    });
  };

  App.drawCompareChart = function (rows) {
    var canvas = App.$('#compare-chart');
    if (!canvas) { return; }
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#070c17';
    ctx.fillRect(0, 0, W, H);

    if (!rows || !rows.length) {
      ctx.fillStyle = '#65779e';
      ctx.font = '13px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'center';
      ctx.fillText('选择至少两个运行进行对比', W / 2, H / 2);
      return;
    }

    var pad = { l: 150, r: 60, t: 28, b: 30 };
    var plotW = W - pad.l - pad.r;
    var plotH = H - pad.t - pad.b;
    var rowH = plotH / rows.length;
    var barH = Math.max(5, Math.min(13, rowH / 3));

    ctx.strokeStyle = 'rgba(35,50,80,0.7)';
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    for (var g = 0; g <= 4; g += 1) {
      var x = pad.l + (plotW * g) / 4;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + plotH);
      ctx.stroke();
      ctx.fillStyle = '#65779e';
      ctx.textAlign = 'center';
      ctx.fillText((g / 4).toFixed(2), x, H - 10);
    }

    rows.forEach(function (row, index) {
      var center = pad.t + rowH * index + rowH / 2;
      var frame = row.metrics && row.metrics.frame_auc;
      var clip = row.metrics && row.metrics.clip_auc;
      ctx.fillStyle = '#cfd9ef';
      ctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
      ctx.textAlign = 'right';
      var label = row.id.length > 20 ? row.id.slice(0, 19) + '…' : row.id;
      ctx.fillText(label, pad.l - 10, center + 3);

      if (frame !== null && frame !== undefined) {
        var w1 = Math.max(1, Math.min(1, Number(frame)) * plotW);
        ctx.fillStyle = 'rgba(94,234,212,0.82)';
        ctx.fillRect(pad.l, center - barH - 1, w1, barH);
        ctx.fillStyle = '#5eead4';
        ctx.textAlign = 'left';
        ctx.fillText(Number(frame).toFixed(3), pad.l + w1 + 5, center - 1);
      }
      if (clip !== null && clip !== undefined) {
        var w2 = Math.max(1, Math.min(1, Number(clip)) * plotW);
        ctx.fillStyle = 'rgba(129,140,248,0.82)';
        ctx.fillRect(pad.l, center + 1, w2, barH);
        ctx.fillStyle = '#a5b4fc';
        ctx.textAlign = 'left';
        ctx.fillText(Number(clip).toFixed(3), pad.l + w2 + 5, center + barH + 1);
      }
    });

    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.fillStyle = '#5eead4';
    ctx.fillRect(pad.l, 8, 8, 8);
    ctx.fillStyle = '#91a2c7';
    ctx.textAlign = 'left';
    ctx.fillText('Frame AUC', pad.l + 12, 16);
    ctx.fillStyle = '#a5b4fc';
    ctx.fillRect(pad.l + 100, 8, 8, 8);
    ctx.fillStyle = '#91a2c7';
    ctx.fillText('Clip AUC', pad.l + 112, 16);
  };
})();
