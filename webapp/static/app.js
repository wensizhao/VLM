(function () {
  'use strict';

  var App = window.App;

  App.loadHealth = async function () {
    try {
      var health = await App.api('/api/health');
      App.state.health = health;
      var dot = App.$('#device-dot');
      var label = App.$('#device-label');
      var device = health.device || {};
      if (dot) {
        dot.className = 'dot ' + (device.cuda ? 'ok' : 'warn');
      }
      if (label) {
        label.textContent = device.cuda
          ? 'CUDA · ' + (device.gpu_name || 'GPU')
          : 'CPU 模式' + (device.torch ? ' · torch ' + device.torch : '');
      }
      var path = App.$('#project-path');
      if (path) {
        path.textContent = health.project_root || '';
        path.title = health.project_root || '';
      }
      return health;
    } catch (error) {
      App.toast('健康检查失败：' + error.message, 'error');
      return null;
    }
  };

  App.loadOverview = async function () {
    try {
      var data = await App.api('/api/overview');
      App.state.datasets = data.datasets || [];
      App.state.runs = data.runs || [];
      App.renderOverviewStats(data);
      App.renderRecentRuns(data.recent_runs || []);
      App.renderSystemStatus(data);
      App.setActiveJob(data.active_job);
      if (App.renderDatasetList) { App.renderDatasetList(); }
      if (App.populateTrainDatasets) { App.populateTrainDatasets(); }
    } catch (error) {
      App.toast('概览加载失败：' + error.message, 'error');
    }
  };

  App.renderOverviewStats = function (data) {
    var host = App.$('#overview-stats');
    if (!host) { return; }
    var datasets = data.datasets || [];
    var runs = data.runs || [];
    var best = data.best_run;
    var bestAuc = best && best.metrics ? best.metrics.frame_auc : null;
    var readyCount = datasets.filter(function (item) { return item.ready; }).length;
    var device = (App.state.health || {}).device || {};
    var cards = [
      App.metricCard('数据集', String(datasets.length), readyCount + ' 个可训练', ''),
      App.metricCard('运行结果', String(runs.length), '训练 / zero-shot', ''),
      App.metricCard('最佳 Frame AUC', App.fmt(bestAuc, 4), best ? ((data.best_scope === 'official' ? '官方标签 · ' : '含历史/伪标签 · ') + best.id) : '还没有结果', 'accent'),
      App.metricCard('当前设备', device.cuda ? 'CUDA' : 'CPU', device.gpu_name || (device.torch ? 'torch ' + device.torch : '未检测到 GPU'), device.cuda ? 'accent' : 'warn')
    ];
    host.innerHTML = cards.join('');
  };

  App.renderRecentRuns = function (runs) {
    var host = App.$('#recent-runs');
    if (!host) { return; }
    if (!runs.length) {
      host.innerHTML = App.empty('◎', '还没有运行结果，先去训练或跑一次 zero-shot。');
      return;
    }
    host.innerHTML = runs.map(function (run) {
      var metrics = run.metrics || {};
      return '<div class="list-row" data-action="open-run" data-id="' + App.escape(run.id) + '">'
        + '<div class="list-main">'
        + '<div class="list-title">' + App.escape(run.id) + '</div>'
        + '<div class="list-sub">' + App.kindLabel(run.kind) + ' · ' + App.escape(run.label_mode || 'unknown') + ' · ' + App.ago(run.modified) + '</div>'
        + '</div>'
        + '<div class="list-metric">Frame AUC<br><span class="' + App.aucClass(metrics.frame_auc) + '">' + App.fmt(metrics.frame_auc, 4) + '</span></div>'
        + '</div>';
    }).join('');
  };

  App.renderSystemStatus = function (data) {
    var host = App.$('#system-status');
    if (!host) { return; }
    var health = App.state.health || {};
    var device = health.device || {};
    var datasets = data.datasets || [];
    var featureReady = datasets.filter(function (item) { return item.feature_ready; }).length;
    var rows = [
      ['设备', device.cuda ? 'CUDA · ' + (device.gpu_name || 'GPU') : 'CPU'],
      ['PyTorch', device.torch || '未检测到'],
      ['官方标签', health.official_labels ? '可用' : '不可用'],
      ['特征缓存', featureReady + ' 个数据集可用'],
      ['项目根目录', health.project_root || '-']
    ];
    host.innerHTML = '<div class="kv">' + rows.map(function (row) {
      return '<div class="kv-row"><span class="kv-key">' + App.escape(row[0]) + '</span>'
        + '<span class="kv-value">' + App.escape(row[1]) + '</span></div>';
    }).join('') + '</div>';
  };

  App.bindNavigation = function () {
    App.$$('.nav-item').forEach(function (item) {
      item.addEventListener('click', function () {
        App.showView(item.getAttribute('data-view'));
      });
    });
    document.addEventListener('click', function (event) {
      var goto = event.target.closest('[data-goto]');
      if (goto) {
        App.showView(goto.getAttribute('data-goto'));
      }
    });
    var refresh = App.$('#refresh-btn');
    if (refresh) {
      refresh.addEventListener('click', async function () {
        await App.loadHealth();
        if (App.state.currentView === 'overview') { App.loadOverview(); }
        else if (App.state.currentView === 'datasets') { App.loadDatasets(true); }
        else if (App.state.currentView === 'validate') { App.loadRuns(); App.loadJobs(); }
        else if (App.state.currentView === 'compare') { App.loadCompare(); }
        App.toast('已刷新');
      });
    }
    window.addEventListener('hashchange', function () {
      var name = (location.hash || '#overview').slice(1);
      if (App.VIEW_META[name] && name !== App.state.currentView) {
        App.showView(name);
      }
    });
  };

  App.bindGlobalActions = function () {
    document.addEventListener('click', function (event) {
      var target = event.target.closest('[data-action]');
      if (target) {
        var action = target.getAttribute('data-action');
        if (action === 'dataset-select' || action === 'refresh-dataset') {
          App.loadDatasetDetail(target.getAttribute('data-id'));
        } else if (action === 'prepare-dataset') {
          App.prepareDataset(target.getAttribute('data-id'));
        } else if (action === 'preview-video') {
          App.startPreview(
            target.getAttribute('data-id'),
            target.getAttribute('data-split'),
            target.getAttribute('data-video'),
            target.getAttribute('data-frames')
          );
        } else if (action === 'close-preview') {
          App.stopPreview();
        } else if (action === 'zoom') {
          App.zoom(target.getAttribute('data-src'), target.getAttribute('data-caption'));
        } else if (action === 'open-run') {
          App.openRun(target.getAttribute('data-id'));
        } else if (action === 'job-log') {
          App.openJobLog(target.getAttribute('data-id'));
        } else if (action === 'job-stop') {
          App.stopJobById(target.getAttribute('data-id'));
        }
        return;
      }
      if (event.target.closest('[data-close-modal]')) {
        App.closeModal();
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') { App.closeModal(); }
    });
    var modal = App.$('#modal');
    if (modal) {
      modal.addEventListener('click', function (event) {
        if (event.target === modal) { App.closeModal(); }
      });
    }
  };

  App.detectActiveJob = async function () {
    try {
      var data = await App.api('/api/jobs');
      var active = data.active;
      if (active) {
        App.attachJob(active.id, {
          title: App.kindLabel(active.kind),
          kind: active.kind,
          runName: active.run_name,
          resultRun: (active.meta || {}).result_run || (active.meta || {}).result_id,
          onDone: function () { App.loadRuns(); }
        });
        App.toast('检测到运行中的任务：' + active.run_name);
      }
    } catch (error) {
      // no active job or endpoint unavailable
    }
  };

  async function init() {
    App.bindNavigation();
    App.bindGlobalActions();
    App.setupUpload();
    App.setupTrain();
    App.setupValidate();
    App.setupCompare();
    await App.loadHealth();
    var initial = (location.hash || '#overview').slice(1);
    if (!App.VIEW_META[initial]) { initial = 'overview'; }
    App.showView(initial);
    App.detectActiveJob();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
