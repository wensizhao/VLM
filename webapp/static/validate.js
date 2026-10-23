(function () {
  'use strict';

  var App = window.App;

  App.loadRuns = async function (selectId) {
    try {
      var data = await App.api('/api/runs');
      App.state.runs = data.runs || [];
      App.renderRunSelect(selectId);
      App.renderCompareRuns();
      var defaultRun = (App.state.runs || []).filter(function (run) { return run.has_checkpoint; })[0]
        || (App.state.runs[0] || null);
      var target = selectId || App.state.pendingRun
        || (App.state.currentRun && App.state.currentRun.id)
        || (defaultRun && defaultRun.id);
      App.state.pendingRun = null;
      if (target) {
        var select = App.$('#validate-run');
        if (select) { select.value = target; }
        await App.loadRunDetail(target);
      } else {
        var empty = App.$('#validate-empty');
        var content = App.$('#validate-content');
        if (empty) { empty.classList.remove('hidden'); }
        if (content) { content.classList.add('hidden'); }
      }
    } catch (error) {
      App.toast('运行列表加载失败：' + error.message, 'error');
    }
  };

  App.renderRunSelect = function (selectId) {
    var select = App.$('#validate-run');
    if (!select) { return; }
    var previous = selectId || select.value;
    var options = ['<option value="">选择运行…</option>'];
    (App.state.runs || []).forEach(function (run) {
      var auc = (run.metrics || {}).frame_auc;
      var label = run.id + ' · ' + App.kindLabel(run.kind) + (auc === null || auc === undefined ? '' : ' · ' + Number(auc).toFixed(3));
      options.push('<option value="' + App.escape(run.id) + '">' + App.escape(label) + '</option>');
    });
    select.innerHTML = options.join('');
    if (previous) { select.value = previous; }
  };

  App.openRun = function (runId) {
    App.state.pendingRun = runId;
    App.showView('validate');
    App.loadRuns(runId);
  };

  App.loadRunDetail = async function (runId) {
    if (!runId) { return; }
    try {
      var data = await App.api('/api/runs/detail?id=' + App.encode(runId));
      App.state.currentRun = data;
      App.renderRunDetail(data);
    } catch (error) {
      App.toast('运行详情加载失败：' + error.message, 'error');
    }
  };

  App.renderRunDetail = function (data) {
    var empty = App.$('#validate-empty');
    var content = App.$('#validate-content');
    if (empty) { empty.classList.add('hidden'); }
    if (content) { content.classList.remove('hidden'); }

    var header = App.$('#validate-header');
    if (header) {
      var checkpointBadge = data.has_checkpoint
        ? '<span class="badge ok">checkpoint</span>'
        : '<span class="badge warn">无 checkpoint</span>';
      var labelBadge = data.label_mode
        ? '<span class="badge">' + App.escape(data.label_mode) + '</span>'
        : '';
      var validBadge = data.benchmark_valid
        ? '<span class="badge ok">官方标签</span>'
        : '<span class="badge warn">非官方标签</span>';
      header.innerHTML = '<span class="run-title">' + App.escape(data.id) + '</span>'
        + '<span class="badge info">' + App.escape(App.kindLabel(data.kind)) + '</span>'
        + labelBadge + validBadge + checkpointBadge
        + '<span class="hint">' + App.fmtDate(data.modified) + '</span>';
    }

    var metrics = data.metrics || {};
    App.renderMetrics(metrics);
    App.renderPerVideo(data.per_video || {});
    App.renderGallery(data.plots || []);
    App.renderExplanationOptions(data);
    App.renderConfig(data);
    App.renderDemo(data.demo || []);

    var motion = App.$('#eval-motion');
    var official = App.$('#eval-official');
    var demo = App.$('#run-demo');
    if (motion) { motion.disabled = !data.has_checkpoint; }
    if (official) { official.disabled = !data.official_available; }
    if (demo) { demo.disabled = !data.has_checkpoint; }

    var notes = App.$('#validate-notes');
    if (notes) {
      var protocol = data.training_protocol || {};
      var lines = [];
      lines.push('label_mode: ' + App.escape(data.label_mode || 'unknown'));
      lines.push('benchmark_valid: ' + (data.benchmark_valid ? 'true' : 'false'));
      if (protocol.mode) { lines.push('training_protocol: ' + App.escape(protocol.mode)); }
      lines.push('指标协议：pooled raw frame scores；不做逐视频归一化。');
      lines.push('单类 AUC 输出 null，不伪装成 0 分。');
      notes.innerHTML = lines.join('<br>');
    }
  };

  App.renderMetrics = function (metrics) {
    var host = App.$('#validate-metrics');
    if (!host) { return; }
    var cards = [
      ['Frame AUC', metrics.frame_auc],
      ['Frame AP', metrics.frame_ap],
      ['Clip AUC', metrics.clip_auc],
      ['Clip AP', metrics.clip_ap],
      ['Video AUC', metrics.video_auc],
      ['Video AP', metrics.video_ap]
    ].map(function (entry) {
      var value = entry[1];
      var html = '<span class="' + App.aucClass(value) + '">' + App.fmt(value, 4) + '</span>';
      return App.metricCard(entry[0], html, null, '');
    });
    host.innerHTML = cards.join('');
  };

  App.renderPerVideo = function (perVideo) {
    var host = App.$('#validate-table');
    if (!host) { return; }
    var ids = Object.keys(perVideo || {}).sort();
    if (!ids.length) {
      host.innerHTML = App.empty('·', '没有逐视频指标');
      return;
    }
    var rows = ids.map(function (id) {
      var item = perVideo[id] || {};
      var auc = item.frame_auc;
      return [
        '<span class="num">' + App.escape(id) + '</span>',
        '<span class="num">' + App.fmtInt(item.num_frames) + '</span>',
        '<span class="num">' + App.fmtInt(item.num_anomaly_frames) + '</span>',
        '<span class="num ' + App.aucClass(auc) + '">' + App.fmt(auc) + '</span>',
        '<span class="num">' + App.fmt(item.frame_ap) + '</span>'
      ];
    });
    host.innerHTML = App.table(['Video', 'Frames', 'Anomaly frames', 'Frame AUC', 'Frame AP'], rows);
  };

  App.renderGallery = function (plots) {
    var host = App.$('#validate-gallery');
    var count = App.$('#plot-count');
    if (count) { count.textContent = (plots || []).length + ' 张'; }
    if (!host) { return; }
    if (!plots || !plots.length) {
      host.innerHTML = App.empty('▧', '还没有生成图表');
      return;
    }
    host.innerHTML = plots.map(function (plot) {
      return '<div class="gallery-item" data-action="zoom" data-src="' + App.escape(plot.url)
        + '" data-caption="' + App.escape(plot.name) + '">'
        + '<img loading="lazy" src="' + App.escape(plot.url) + '" alt="' + App.escape(plot.name) + '">'
        + '<div class="gallery-caption">' + App.escape(plot.name) + '</div></div>';
    }).join('');
  };

  App.renderExplanationOptions = function (data) {
    var select = App.$('#explain-video');
    if (!select) { return; }
    var keys = Object.keys(data.explanations || {}).sort();
    if (!keys.length) {
      select.innerHTML = '<option value="">无解释</option>';
      var host = App.$('#validate-explanations');
      if (host) { host.innerHTML = App.empty('·', '该运行没有 explanations.json'); }
      return;
    }
    select.innerHTML = keys.map(function (key) {
      return '<option value="' + App.escape(key) + '">' + App.escape(key) + '</option>';
    }).join('');
    App.renderExplanation(data, keys[0]);
  };

  App.renderExplanation = function (data, videoId) {
    var host = App.$('#validate-explanations');
    if (!host) { return; }
    var entry = (data.explanations || {})[videoId];
    if (!entry) {
      host.innerHTML = App.empty('·', '没有该视频的解释');
      return;
    }
    var prompts = entry.top_prompts || [];
    var maxSim = 0.000001;
    prompts.forEach(function (prompt) {
      maxSim = Math.max(maxSim, Math.abs(Number(prompt.similarity) || 0));
    });
    var rows = prompts.slice(0, 8).map(function (prompt) {
      var sim = Number(prompt.similarity) || 0;
      var width = Math.max(2, Math.min(100, Math.abs(sim) / maxSim * 100));
      return '<div class="prompt-row">'
        + '<span class="prompt-name" title="' + App.escape(prompt.prompt) + '">' + App.escape(prompt.prompt) + '</span>'
        + '<span class="prompt-track"><span style="width:' + width + '%"></span></span>'
        + '<span class="prompt-sim">' + sim.toFixed(3) + '</span>'
        + '</div>';
    }).join('');
    host.innerHTML = '<div class="explain-text">' + App.escape(entry.explanation || '无解释文本') + '</div>'
      + '<div class="prompt-list">' + rows + '</div>'
      + '<div class="note-box">start_frame: ' + App.escape(entry.start_frame)
      + ' · clip_score: ' + App.fmt(entry.clip_score, 4)
      + ' · decision: ' + App.escape(entry.decision || 'uncalibrated')
      + (entry.top_abnormal_frames ? '<br>top frames: ' + App.escape((entry.top_abnormal_frames || []).join(', ')) : '')
      + '</div>';
  };

  App.renderConfig = function (data) {
    var host = App.$('#validate-config');
    if (!host) { return; }
    var cfg = data.config || {};
    var model = cfg.model || {};
    var prompt = cfg.prompt || {};
    var rows = [
      ['run_name', cfg.run_name || data.id],
      ['seed', cfg.seed],
      ['device', cfg.device],
      ['prompt types', (prompt.types || []).join(', ')],
      ['fusion', (model.fusion || {}).type],
      ['temporal', (model.temporal || {}).type],
      ['label mode', (cfg.data || {}).frame_label_mode || data.label_mode],
      ['checkpoint', data.checkpoint || '无']
    ];
    host.innerHTML = rows.map(function (row) {
      return '<div class="kv-row"><span class="kv-key">' + App.escape(row[0]) + '</span>'
        + '<span class="kv-value">' + App.escape(row[1] === null || row[1] === undefined ? '-' : row[1]) + '</span></div>';
    }).join('');
  };

  App.renderDemo = function (items) {
    var host = App.$('#validate-demo');
    if (!host) { return; }
    if (!items || !items.length) {
      host.innerHTML = '<span class="muted">暂无 demo 文件。可在上方点击“生成可视化 Demo”。</span>';
      return;
    }
    host.innerHTML = items.map(function (item) {
      return '<a class="demo-item" href="' + App.escape(item.url) + '" target="_blank" rel="noopener">'
        + '<span class="demo-kind">' + App.escape(item.kind) + '</span>'
        + '<span class="demo-name">' + App.escape(item.name) + '</span></a>';
    }).join('');
  };

  App.runEval = async function (mode) {
    var run = App.state.currentRun;
    if (!run) { return; }
    try {
      var result = await App.api('/api/runs/evaluate', {
        method: 'POST',
        body: JSON.stringify({ id: run.id, mode: mode })
      });
      App.toast(mode === 'official' ? '官方标签复评已启动' : '运动标签复评已启动');
      App.showView('train');
      App.attachJob(result.job.id, {
        title: mode === 'official' ? '官方标签复评' : '运动标签复评',
        kind: 'eval',
        runName: result.eval.result_id,
        resultRun: result.eval.result_id,
        onDone: function () { App.loadRuns(result.eval.result_id); }
      });
    } catch (error) {
      App.toast('复评启动失败：' + error.message, 'error');
    }
  };

  App.runZeroShot = async function (prompt) {
    try {
      var result = await App.api('/api/zero-shot', {
        method: 'POST',
        body: JSON.stringify({ prompt: prompt, official: true })
      });
      var note = App.$('#zero-shot-result');
      if (note) { note.textContent = 'Zero-shot ' + prompt + ' 已启动…'; }
      App.showView('train');
      App.attachJob(result.job.id, {
        title: 'Zero-shot · ' + prompt,
        kind: 'zero_shot',
        runName: result.zero_shot.result_id,
        resultRun: result.zero_shot.result_id,
        onDone: function () { App.loadRuns(result.zero_shot.result_id); }
      });
    } catch (error) {
      App.toast('Zero-shot 启动失败：' + error.message, 'error');
    }
  };

  App.runDemo = async function () {
    var run = App.state.currentRun;
    if (!run) { return; }
    var raw = window.prompt('输入要渲染的视频 ID（逗号分隔）', '01,06,10');
    if (!raw) { return; }
    var videos = raw.split(',').map(function (item) { return item.trim(); }).filter(Boolean);
    if (!videos.length) { return; }
    try {
      var result = await App.api('/api/runs/demo', {
        method: 'POST',
        body: JSON.stringify({ id: run.id, videos: videos })
      });
      App.toast('Demo 渲染已启动');
      App.showView('train');
      App.attachJob(result.job.id, {
        title: '可视化 Demo',
        kind: 'demo',
        runName: result.demo.result_id,
        onDone: function () { App.loadRuns(run.id); }
      });
    } catch (error) {
      App.toast('Demo 启动失败：' + error.message, 'error');
    }
  };

  App.loadJobs = async function () {
    try {
      var data = await App.api('/api/jobs');
      App.renderJobs(data.jobs || []);
      App.setActiveJob(data.active);
    } catch (error) {
      // silent
    }
  };

  App.renderJobs = function (jobs) {
    var host = App.$('#jobs-table');
    if (!host) { return; }
    if (!jobs.length) {
      host.innerHTML = App.empty('·', '还没有任务记录（仅当前会话）');
      return;
    }
    var rows = jobs.slice(0, 30).map(function (job) {
      var actions = '<button class="link" data-action="job-log" data-id="' + App.escape(job.id) + '">日志</button>';
      if (job.status === 'running' || job.status === 'queued' || job.status === 'stopping') {
        actions += ' <button class="link danger" data-action="job-stop" data-id="' + App.escape(job.id) + '">停止</button>';
      }
      return [
        App.statusBadge(job.status),
        App.escape(App.kindLabel(job.kind)),
        '<span class="num">' + App.escape(job.run_name) + '</span>',
        '<span class="num">' + App.fmtDuration(job.elapsed_sec) + '</span>',
        actions
      ];
    });
    host.innerHTML = App.table(['状态', '类型', 'Run', '耗时', '操作'], rows);
  };

  App.openJobLog = async function (jobId) {
    try {
      var text = await App.api('/api/jobs/' + jobId + '/log?tail=400');
      var body = typeof text === 'string' ? text : (text.error || '');
      App.openModal('<h3 style="margin-bottom:10px">任务日志</h3><pre class="log" style="max-height:70vh">'
        + App.escape(body) + '</pre>');
    } catch (error) {
      App.toast('日志加载失败：' + error.message, 'error');
    }
  };

  App.stopJobById = async function (jobId) {
    try {
      await App.api('/api/jobs/' + jobId + '/stop', { method: 'POST' });
      App.toast('已请求停止任务');
      App.loadJobs();
    } catch (error) {
      App.toast('停止失败：' + error.message, 'error');
    }
  };

  App.setupValidate = function () {
    var select = App.$('#validate-run');
    if (select) {
      select.addEventListener('change', function () {
        if (select.value) { App.loadRunDetail(select.value); }
      });
    }
    var refresh = App.$('#validate-refresh');
    if (refresh) {
      refresh.addEventListener('click', function () {
        App.loadRuns(select ? select.value : null);
      });
    }
    var jobsRefresh = App.$('#jobs-refresh');
    if (jobsRefresh) { jobsRefresh.addEventListener('click', App.loadJobs); }
    var evalMotion = App.$('#eval-motion');
    if (evalMotion) { evalMotion.addEventListener('click', function () { App.runEval('motion'); }); }
    var evalOfficial = App.$('#eval-official');
    if (evalOfficial) { evalOfficial.addEventListener('click', function () { App.runEval('official'); }); }
    var demo = App.$('#run-demo');
    if (demo) { demo.addEventListener('click', App.runDemo); }
    var explain = App.$('#explain-video');
    if (explain) {
      explain.addEventListener('change', function () {
        if (App.state.currentRun) { App.renderExplanation(App.state.currentRun, explain.value); }
      });
    }
    App.$$('.zero-shot').forEach(function (button) {
      button.addEventListener('click', function () {
        App.runZeroShot(button.getAttribute('data-prompt'));
      });
    });
  };
})();
