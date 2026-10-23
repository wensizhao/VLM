(function () {
  'use strict';

  var App = window.App;

  App.loadCompare = async function () {
    try {
      var data = await App.api('/api/runs');
      App.state.runs = data.runs || [];
      var selected = App.state.compareSelected || {};
      var hasSelection = Object.keys(selected).some(function (key) { return selected[key]; });
      if (!hasSelection) {
        App.state.compareSelected = {};
        (App.state.runs || []).slice(0, 3).forEach(function (run) {
          App.state.compareSelected[run.id] = true;
        });
      }
      App.renderCompareRuns();
      App.renderCompare();
    } catch (error) {
      App.toast('对比数据加载失败：' + error.message, 'error');
    }
  };

  App.renderCompareRuns = function () {
    var host = App.$('#compare-runs');
    if (!host) { return; }
    var runs = App.state.runs || [];
    if (!runs.length) {
      host.innerHTML = App.empty('▥', '还没有运行结果');
      return;
    }
    var selected = App.state.compareSelected || {};
    host.innerHTML = runs.map(function (run) {
      var checked = selected[run.id] ? ' checked' : '';
      var auc = (run.metrics || {}).frame_auc;
      return '<label class="check-row">'
        + '<input type="checkbox" data-compare-id="' + App.escape(run.id) + '"' + checked + '>'
        + '<span class="num" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
        + App.escape(run.id) + '</span>'
        + '<span class="badge">' + App.escape(App.kindLabel(run.kind)) + '</span>'
        + '<span class="num ' + App.aucClass(auc) + '">' + App.fmt(auc, 3) + '</span>'
        + '</label>';
    }).join('');
  };

  App.renderCompare = function () {
    var selected = App.state.compareSelected || {};
    var rows = (App.state.runs || []).filter(function (run) { return selected[run.id]; });
    var tableHost = App.$('#compare-table');
    if (tableHost) {
      if (!rows.length) {
        tableHost.innerHTML = App.empty('▥', '勾选至少一个运行');
      } else {
        var body = rows.map(function (run) {
          var metrics = run.metrics || {};
          return [
            '<span class="num">' + App.escape(run.id) + '</span>',
            App.escape(App.kindLabel(run.kind)),
            App.escape(run.label_mode || '-'),
            '<span class="num ' + App.aucClass(metrics.frame_auc) + '">' + App.fmt(metrics.frame_auc) + '</span>',
            '<span class="num">' + App.fmt(metrics.frame_ap) + '</span>',
            '<span class="num ' + App.aucClass(metrics.clip_auc) + '">' + App.fmt(metrics.clip_auc) + '</span>',
            '<span class="num">' + App.fmt(metrics.clip_ap) + '</span>',
            '<span class="num">' + App.fmt(metrics.video_auc) + '</span>'
          ];
        });
        tableHost.innerHTML = App.table(
          ['Run', '类型', '标签', 'Frame AUC', 'Frame AP', 'Clip AUC', 'Clip AP', 'Video AUC'],
          body
        );
      }
    }
    App.drawCompareChart(rows);
  };

  App.setupCompare = function () {
    var refresh = App.$('#compare-refresh');
    if (refresh) { refresh.addEventListener('click', App.loadCompare); }
    var host = App.$('#compare-runs');
    if (host) {
      host.addEventListener('change', function (event) {
        var input = event.target.closest('[data-compare-id]');
        if (!input) { return; }
        App.state.compareSelected[input.getAttribute('data-compare-id')] = input.checked;
        App.renderCompare();
      });
    }
  };
})();
