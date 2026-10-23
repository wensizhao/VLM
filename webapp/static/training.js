(function () {
  'use strict';

  var App = window.App;

  App.populateTrainDatasets = function () {
    var select = App.$('#train-dataset');
    if (!select) { return; }
    var previous = select.value;
    var datasets = App.state.datasets || [];
    var options = ['<option value="">选择数据集…</option>'];
    datasets.forEach(function (dataset) {
      var training = (dataset.splits || {}).training || { count: 0 };
      var testing = (dataset.splits || {}).testing || { count: 0 };
      var suffix = dataset.ready ? '' : ' · 需准备';
      options.push('<option value="' + App.escape(dataset.id) + '">'
        + App.escape(dataset.name || dataset.id) + ' · ' + training.count + '/' + testing.count
        + suffix + '</option>');
    });
    select.innerHTML = options.join('');
    var exists = datasets.some(function (dataset) { return dataset.id === previous; });
    if (exists) {
      select.value = previous;
    } else {
      var preferred = datasets.filter(function (dataset) { return dataset.id === 'Avenue_Dataset'; })[0]
        || datasets.filter(function (dataset) { return dataset.ready; })[0]
        || datasets[0];
      if (preferred) { select.value = preferred.id; }
    }
    var nameInput = App.$('#train-run-name');
    if (nameInput && !nameInput.value) {
      nameInput.value = App.generateRunName();
    }
    App.updateTrainHint();
  };

  App.generateRunName = function () {
    var date = new Date();
    var pad = function (value) { return (value < 10 ? '0' : '') + value; };
    return 'web_' + date.getFullYear() + pad(date.getMonth() + 1) + pad(date.getDate())
      + '_' + pad(date.getHours()) + pad(date.getMinutes()) + pad(date.getSeconds());
  };

  App.updateTrainHint = function () {
    var select = App.$('#train-dataset');
    var hint = App.$('#train-dataset-hint');
    var submit = App.$('#train-submit');
    if (!select || !hint) { return; }
    var dataset = App.datasetById(select.value);
    if (!dataset) {
      hint.innerHTML = '<span class="badge warn">先选择一个数据集</span>';
      if (submit) { submit.disabled = true; }
      return;
    }
    var parts = [];
    var training = (dataset.splits || {}).training || { count: 0 };
    var testing = (dataset.splits || {}).testing || { count: 0 };
    parts.push('<span class="badge">' + training.count + ' train</span>');
    parts.push('<span class="badge">' + testing.count + ' test</span>');
    if (dataset.feature_ready) {
      parts.push('<span class="badge info">特征缓存</span>');
    } else {
      parts.push('<span class="badge warn">像素路径较慢</span>');
    }
    if (!dataset.has_masks) {
      parts.push('<span class="badge danger">缺少 *_vol，请先准备</span>');
    }
    if (!dataset.has_labels) {
      parts.push('<span class="badge warn">缺少 labels，将在线计算运动标签</span>');
    }
    hint.innerHTML = parts.join(' ');
    if (submit) { submit.disabled = !dataset.ready; }
    var skip = App.$('#train-skip-eval');
    if (skip && testing.count === 0) { skip.checked = true; }
  };

  App.collectTrainPayload = function () {
    var select = App.$('#train-dataset');
    var name = App.$('#train-run-name').value.trim();
    if (!select || !select.value) { throw new Error('请选择数据集'); }
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) {
      throw new Error('Run 名称只能包含字母、数字、点、横线和下划线，且不超过 64 字符');
    }
    var promptTypes = App.$$('.prompt-type:checked').map(function (item) { return item.value; });
    if (!promptTypes.length) { throw new Error('至少选择一种 prompt 类型'); }
    return {
      dataset_id: select.value,
      run_name: name,
      epochs: parseInt(App.$('#train-epochs').value, 10) || 10,
      batch_size: parseInt(App.$('#train-batch').value, 10) || 4,
      device: App.$('#train-device').value,
      fusion: App.$('#train-fusion').value,
      temporal: App.$('#train-temporal').value,
      label_mode: App.$('#train-label-mode').value,
      use_feature: App.$('#train-use-feature').value,
      prompt_types: promptTypes,
      skip_eval: App.$('#train-skip-eval').checked,
      num_workers: 0
    };
  };

  App.submitTrain = async function (event) {
    if (event) { event.preventDefault(); }
    var error = App.$('#train-form-error');
    var button = App.$('#train-submit');
    if (error) { error.textContent = ''; }
    var payload;
    try {
      payload = App.collectTrainPayload();
    } catch (err) {
      if (error) { error.textContent = err.message; }
      return;
    }
    if (button) { button.disabled = true; }
    try {
      var result = await App.api('/api/train', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      App.toast('训练已启动：' + result.run.run_name);
      App.attachJob(result.job.id, {
        title: '训练',
        kind: 'train',
        runName: result.run.run_name,
        resultRun: result.run.run_name,
        epochs: result.run.epochs,
        onDone: function () {
          App.loadRuns(result.run.run_name);
        }
      });
    } catch (err) {
      if (error) { error.textContent = err.message; }
      App.toast('启动失败：' + err.message, 'error');
    } finally {
      if (button) { button.disabled = false; }
    }
  };

  App.attachJob = function (jobId, options) {
    var opts = options || {};
    App.state.jobOptions = opts;
    App.state.activeJobId = jobId;
    if (App.state.eventSource) {
      App.state.eventSource.close();
      App.state.eventSource = null;
    }
    var empty = App.$('#job-empty');
    var live = App.$('#job-live');
    if (empty) { empty.classList.add('hidden'); }
    if (live) { live.classList.remove('hidden'); }
    var title = App.$('#job-title');
    if (title) { title.textContent = opts.title || '实时进度'; }
    var status = App.$('#job-status');
    if (status) { status.className = 'badge info'; status.textContent = '连接中'; }
    var results = App.$('#job-results');
    if (results) { results.classList.add('hidden'); }
    App.setActiveJob({ status: 'running', kind: opts.kind || 'train', run_name: opts.runName || jobId });

    App.api('/api/jobs/' + jobId).then(function (job) {
      App.renderJob(job, opts);
    }).catch(function () {});

    var source = new EventSource('/api/jobs/' + jobId + '/events');
    App.state.eventSource = source;
    source.onmessage = function (event) {
      var job;
      try { job = JSON.parse(event.data); } catch (error) { return; }
      if (job.error && !job.id) {
        App.toast(job.error, 'error');
        return;
      }
      App.renderJob(job, opts);
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
        source.close();
        App.state.eventSource = null;
        App.finishJob(job, opts);
      }
    };
    source.onerror = function () {
      source.close();
      App.state.eventSource = null;
      App.pollJob(jobId, opts, 0);
    };
  };

  App.pollJob = function (jobId, opts, attempt) {
    var tries = attempt || 0;
    if (tries > 600) { return; }
    setTimeout(async function () {
      try {
        var job = await App.api('/api/jobs/' + jobId);
        App.renderJob(job, opts);
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
          App.finishJob(job, opts);
          return;
        }
      } catch (error) {
        // keep polling
      }
      App.pollJob(jobId, opts, tries + 1);
    }, 1600);
  };

  App.finishJob = function (job, opts) {
    var options = opts || {};
    App.setActiveJob(null);
    if (App.state.lastTerminalJob !== job.id) {
      App.state.lastTerminalJob = job.id;
      if (job.status === 'completed') {
        App.toast(options.title ? (options.title + '完成') : '任务完成', 'ok');
      } else if (job.status === 'stopped') {
        App.toast('任务已停止');
      } else {
        App.toast((options.title || '任务') + '失败：' + (job.error || '未知错误'), 'error');
      }
    }
    if (options.onDone) {
      try { options.onDone(job); } catch (error) { /* ignore */ }
    }
    if (App.loadRuns) { App.loadRuns(options.resultRun || null); }
    if (App.loadJobs) { App.loadJobs(); }
  };

  App.stopJob = async function () {
    var jobId = App.state.activeJobId;
    if (!jobId) { return; }
    try {
      var result = await App.api('/api/jobs/' + jobId + '/stop', { method: 'POST' });
      App.renderJob(result.job, App.state.jobOptions || {});
      App.toast('已请求停止任务');
    } catch (error) {
      App.toast('停止失败：' + error.message, 'error');
    }
  };

  App.renderJob = function (job, opts) {
    var options = opts || {};
    var empty = App.$('#job-empty');
    var live = App.$('#job-live');
    if (empty) { empty.classList.add('hidden'); }
    if (live) { live.classList.remove('hidden'); }
    var title = App.$('#job-title');
    if (title) { title.textContent = options.title || App.kindLabel(job.kind); }
    var status = App.$('#job-status');
    if (status) {
      status.outerHTML = App.statusBadge(job.status).replace('class="badge', 'id="job-status" class="badge');
    }
    var run = App.$('#job-run');
    if (run) { run.textContent = App.kindLabel(job.kind) + ' · ' + (job.run_name || job.id); }
    var elapsed = App.$('#job-elapsed');
    if (elapsed) { elapsed.textContent = App.fmtDuration(job.elapsed_sec); }

    var progress = job.progress || {};
    var total = Number((job.meta || {}).epochs || options.epochs || 0);
    var epoch = Number(progress.epoch || 0);
    var percent = Number(progress.percent || 0);
    var overall = percent;
    if (total && epoch) {
      overall = Math.min(99, ((epoch - 1) + percent / 100) / total * 100);
    }
    if (progress.phase === 'test') { overall = 99; }
    if (job.status === 'completed') { overall = 100; }
    var bar = App.$('#job-bar');
    if (bar) { bar.style.width = Math.max(0, Math.min(100, overall)) + '%'; }
    var percentNode = App.$('#job-percent');
    if (percentNode) { percentNode.textContent = Math.round(overall) + '%'; }

    var phase = App.$('#job-phase');
    if (phase) {
      var phaseLabel = {
        train: '训练', val: '验证', test: '测试评估', prepare: '数据准备', finished: '收尾'
      }[progress.phase] || '等待输出';
      var text = phaseLabel;
      if (epoch && total) { text += ' · epoch ' + epoch + ' / ' + total; }
      if (progress.message) { text += ' · ' + progress.message; }
      if (job.status === 'failed' && job.error) { text += ' · ' + job.error; }
      phase.textContent = text;
    }

    var metricHost = App.$('#job-metrics');
    if (metricHost) {
      var metrics = job.metrics || {};
      var cards = [];
      if (metrics.train) {
        cards.push(App.metricCard('train loss', App.fmt(metrics.train.loss, 4), null, 'accent'));
        cards.push(App.metricCard('BCE', App.fmt(metrics.train.bce, 4), null));
        cards.push(App.metricCard('contrastive', App.fmt(metrics.train.contrastive, 4), null));
      }
      if (metrics.val) {
        cards.push(App.metricCard('val frame AUC', App.fmt(metrics.val.frame_auc, 4), null, 'accent'));
        cards.push(App.metricCard('val clip AUC', App.fmt(metrics.val.clip_auc, 4), null));
      }
      if (metrics.test) {
        cards.push(App.metricCard('test frame AUC', App.fmt(metrics.test.frame_auc, 4), null, 'accent'));
        cards.push(App.metricCard('test frame AP', App.fmt(metrics.test.frame_ap, 4), null));
      }
      if (metrics.zero_shot) {
        cards.push(App.metricCard('zero-shot frame AUC', App.fmt(metrics.zero_shot.frame_auc, 4), null, 'accent'));
        cards.push(App.metricCard('zero-shot frame AP', App.fmt(metrics.zero_shot.frame_ap, 4), null));
      }
      metricHost.innerHTML = cards.slice(-8).join('');
    }

    App.drawTrainingChart((job.metrics || {}).history || []);

    var log = App.$('#train-log');
    if (log) {
      var nearBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 36;
      log.textContent = (job.log_tail || []).join('\n');
      if (nearBottom) { log.scrollTop = log.scrollHeight; }
    }

    var stop = App.$('#job-stop');
    if (stop) {
      stop.disabled = job.status === 'completed' || job.status === 'failed' || job.status === 'stopped';
      stop.textContent = job.status === 'stopping' ? '停止中…' : '停止任务';
    }

    var resultRun = options.resultRun || (job.meta || {}).result_run || (job.meta || {}).result_id;
    var results = App.$('#job-results');
    if (results && resultRun) {
      results.classList.remove('hidden');
      results.setAttribute('data-run', resultRun);
    }
    App.setActiveJob(job);
  };

  App.setupTrain = function () {
    var form = App.$('#train-form');
    if (form) {
      form.addEventListener('submit', App.submitTrain);
    }
    var select = App.$('#train-dataset');
    if (select) {
      select.addEventListener('change', App.updateTrainHint);
    }
    var stop = App.$('#job-stop');
    if (stop) {
      stop.addEventListener('click', App.stopJob);
    }
    var results = App.$('#job-results');
    if (results) {
      results.addEventListener('click', function () {
        var run = results.getAttribute('data-run');
        if (run && App.openRun) { App.openRun(run); }
      });
    }
  };
})();
