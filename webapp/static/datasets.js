(function () {
  'use strict';

  var App = window.App;

  App.datasetById = function (id) {
    return (App.state.datasets || []).filter(function (item) { return item.id === id; })[0] || null;
  };

  App.loadDatasets = async function (force) {
    try {
      var data = await App.api('/api/datasets');
      App.state.datasets = data.datasets || [];
      App.renderDatasetList();
      if (App.populateTrainDatasets) {
        App.populateTrainDatasets();
      }
    } catch (error) {
      App.toast('数据集加载失败：' + error.message, 'error');
    }
  };

  App.renderDatasetList = function () {
    var host = App.$('#dataset-list');
    var count = App.$('#dataset-count');
    var datasets = App.state.datasets || [];
    if (count) {
      count.textContent = datasets.length + ' 个';
    }
    if (!host) {
      return;
    }
    if (!datasets.length) {
      host.innerHTML = App.empty('▤', '还没有数据集。拖入一个 Avenue 结构文件夹，或注册服务器路径。');
      return;
    }
    host.innerHTML = datasets.map(function (dataset) {
      var splits = dataset.splits || {};
      var training = splits.training || { count: 0 };
      var testing = splits.testing || { count: 0 };
      var badges = [];
      if (dataset.ready) { badges.push('<span class="badge ok">可训练</span>'); }
      if (dataset.needs_prep) { badges.push('<span class="badge warn">需准备</span>'); }
      if (dataset.feature_ready) { badges.push('<span class="badge info">特征缓存</span>'); }
      if (dataset.official_ready) { badges.push('<span class="badge">官方标签</span>'); }
      var active = App.state.currentDataset && App.state.currentDataset.id === dataset.id ? ' active' : '';
      return '<div class="dataset-card' + active + '" data-action="dataset-select" data-id="' + App.escape(dataset.id) + '">'
        + '<div class="dataset-card-top"><span class="dataset-name">' + App.escape(dataset.name || dataset.id) + '</span>'
        + '<span class="hint">' + training.count + ' train / ' + testing.count + ' test</span></div>'
        + '<div class="dataset-path">' + App.escape(dataset.path || dataset.id) + '</div>'
        + '<div class="badge-row">' + badges.join('') + '</div>'
        + '</div>';
    }).join('');
  };

  App.loadDatasetDetail = async function (id) {
    try {
      var data = await App.api('/api/datasets/detail?id=' + App.encode(id));
      App.state.currentDataset = data;
      App.renderDatasetList();
      App.renderDatasetDetail(data);
    } catch (error) {
      App.toast('数据集详情加载失败：' + error.message, 'error');
    }
  };

  App.renderDatasetDetail = function (data) {
    var host = App.$('#dataset-detail');
    if (!host) { return; }
    host.classList.remove('hidden');
    var splits = data.splits || {};
    var training = splits.training || { count: 0, frames: 0 };
    var testing = splits.testing || { count: 0, frames: 0 };
    var badges = [];
    if (data.ready) { badges.push('<span class="badge ok">可训练</span>'); }
    if (data.needs_prep) { badges.push('<span class="badge warn">需要预处理</span>'); }
    if (data.feature_ready) { badges.push('<span class="badge info">特征缓存</span>'); }
    if (data.official_ready) { badges.push('<span class="badge">官方标签可用</span>'); }
    if (!data.has_masks) { badges.push('<span class="badge danger">缺少 *_vol</span>'); }
    if (!data.has_labels) { badges.push('<span class="badge warn">缺少 labels</span>'); }

    var html = '<div class="dataset-hero">'
      + '<div><h2>' + App.escape(data.name || data.id) + '</h2>'
      + '<div class="hint">' + App.escape(data.path || '') + '</div>'
      + '<div class="badge-row" style="margin-top:8px">' + badges.join('') + '</div></div>'
      + '<div class="dataset-actions">'
      + '<button class="btn btn-primary" data-action="prepare-dataset" data-id="' + App.escape(data.id) + '">一键准备</button>'
      + '<button class="btn" data-action="refresh-dataset" data-id="' + App.escape(data.id) + '">刷新</button>'
      + '</div></div>'
      + '<div class="prep-options">'
      + '<label class="check"><input id="prep-labels" type="checkbox" checked> 生成运动标签</label>'
      + '<label class="check"><input id="prep-features" type="checkbox"> 提取 CLIP 特征（慢，但训练快 10×）</label>'
      + '<span class="hint">' + training.count + ' 训练视频 · ' + testing.count + ' 测试视频 · '
      + App.fmtInt(training.frames + testing.frames) + ' 帧</span>'
      + '</div>'
      + '<div id="preview-slot"></div>';

    ['training', 'testing'].forEach(function (split) {
      var videos = (data.videos || []).filter(function (video) { return video.split === split; });
      var meta = splits[split] || { count: 0, frames: 0 };
      html += '<div style="margin-top:18px">'
        + '<div class="card-head"><h3>' + split + '</h3><span class="hint">'
        + meta.count + ' videos · ' + App.fmtInt(meta.frames) + ' frames</span></div>';
      if (!videos.length) {
        html += App.empty('·', '该 split 没有视频');
      } else {
        html += '<div class="video-grid">' + videos.map(function (video) {
          var label = video.label;
          var labelText = '无标签';
          if (label) {
            labelText = label.source + ' · ' + (label.anomaly_frames === null ? '?' : label.anomaly_frames) + ' anomaly';
          }
          return '<div class="video-card" data-action="preview-video" data-id="' + App.escape(data.id)
            + '" data-split="' + split + '" data-video="' + App.escape(video.id)
            + '" data-frames="' + App.fmtInt(video.num_frames) + '">'
            + '<img loading="lazy" src="' + App.escape(video.thumbnail) + '" alt="' + App.escape(video.id) + '">'
            + '<div class="video-card-body">'
            + '<div class="video-id">' + App.escape(video.id) + '</div>'
            + '<div class="video-meta">' + App.fmtInt(video.num_frames) + ' frames · ' + video.width + '×' + video.height + '</div>'
            + '<div class="video-label">' + App.escape(labelText) + '</div>'
            + '</div></div>';
        }).join('') + '</div>';
      }
      html += '</div>';
    });
    host.innerHTML = html;
  };

  App.startPreview = function (id, split, video, frames) {
    App.stopPreview();
    var total = Math.max(1, parseInt(frames, 10) || 1);
    App.state.preview = { id: id, split: split, video: video, frames: total, index: 0, playing: false };
    var slot = App.$('#preview-slot');
    if (!slot) { return; }
    slot.innerHTML = '<div class="preview-panel">'
      + '<div class="preview-frame"><img id="preview-img" alt="preview"></div>'
      + '<div class="preview-controls">'
      + '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
      + '<strong>' + App.escape(video) + '</strong>'
      + '<span class="badge">' + App.escape(split) + '</span></div>'
      + '<input id="preview-slider" type="range" min="0" max="' + (total - 1) + '" value="0">'
      + '<div class="preview-meta" id="preview-meta"></div>'
      + '<div class="preview-actions">'
      + '<button class="btn" id="preview-play" type="button">播放</button>'
      + '<button class="btn btn-ghost" data-action="close-preview" type="button">关闭</button>'
      + '</div></div></div>';
    var slider = App.$('#preview-slider');
    if (slider) {
      slider.addEventListener('input', function () {
        App.setPreviewFrame(parseInt(slider.value, 10) || 0);
      });
    }
    var play = App.$('#preview-play');
    if (play) {
      play.addEventListener('click', function () {
        App.togglePreviewPlay();
      });
    }
    App.setPreviewFrame(0);
  };

  App.setPreviewFrame = function (index) {
    var preview = App.state.preview;
    if (!preview) { return; }
    var safe = Math.max(0, Math.min(index, preview.frames - 1));
    preview.index = safe;
    var img = App.$('#preview-img');
    var slider = App.$('#preview-slider');
    var meta = App.$('#preview-meta');
    if (img) {
      img.src = '/api/datasets/frame?id=' + App.encode(preview.id)
        + '&split=' + preview.split + '&video=' + App.encode(preview.video)
        + '&index=' + safe + '&w=960&_=' + Date.now();
    }
    if (slider) { slider.value = String(safe); }
    if (meta) {
      meta.textContent = preview.video + ' · frame ' + safe + ' / ' + (preview.frames - 1);
    }
  };

  App.togglePreviewPlay = function () {
    var preview = App.state.preview;
    if (!preview) { return; }
    var button = App.$('#preview-play');
    if (preview.playing) {
      preview.playing = false;
      if (App.state.previewTimer) {
        clearInterval(App.state.previewTimer);
        App.state.previewTimer = null;
      }
      if (button) { button.textContent = '播放'; }
      return;
    }
    preview.playing = true;
    if (button) { button.textContent = '暂停'; }
    App.state.previewTimer = setInterval(function () {
      if (!App.state.preview) { return; }
      var next = App.state.preview.index + 1;
      if (next >= App.state.preview.frames) { next = 0; }
      App.setPreviewFrame(next);
    }, 130);
  };

  App.stopPreview = function () {
    if (App.state.previewTimer) {
      clearInterval(App.state.previewTimer);
      App.state.previewTimer = null;
    }
    App.state.preview = null;
    var slot = App.$('#preview-slot');
    if (slot) { slot.innerHTML = ''; }
  };

  App.prepareDataset = async function (id) {
    var labels = App.$('#prep-labels') ? App.$('#prep-labels').checked : true;
    var features = App.$('#prep-features') ? App.$('#prep-features').checked : false;
    try {
      var result = await App.api('/api/datasets/prepare', {
        method: 'POST',
        body: JSON.stringify({ id: id, labels: labels, features: features, device: 'auto' })
      });
      App.toast('数据准备任务已启动');
      App.showView('train');
      App.attachJob(result.job.id, {
        title: '数据准备',
        runName: result.job.run_name,
        onDone: function () {
          App.loadDatasets(true);
          if (App.state.currentDataset) { App.loadDatasetDetail(App.state.currentDataset.id); }
        }
      });
    } catch (error) {
      App.toast('准备失败：' + error.message, 'error');
    }
  };

  App.setupUpload = function () {
    var folderInput = App.$('#folder-input');
    var zipInput = App.$('#zip-input');
    var pickFolder = App.$('#pick-folder');
    var pickZip = App.$('#pick-zip');
    var pickZip2 = App.$('#pick-zip-2');
    var dropzone = App.$('#dropzone');

    if (pickFolder && folderInput) {
      pickFolder.addEventListener('click', function () { folderInput.click(); });
    }
    if (pickZip && zipInput) {
      pickZip.addEventListener('click', function () { zipInput.click(); });
    }
    if (pickZip2 && zipInput) {
      pickZip2.addEventListener('click', function () { zipInput.click(); });
    }
    if (folderInput) {
      folderInput.addEventListener('change', function () {
        var files = Array.prototype.slice.call(folderInput.files || []);
        var items = files.map(function (file) {
          return { file: file, path: file.webkitRelativePath || file.name };
        });
        App.uploadItems(items);
        folderInput.value = '';
      });
    }
    if (zipInput) {
      zipInput.addEventListener('change', function () {
        var files = Array.prototype.slice.call(zipInput.files || []);
        var items = files.map(function (file) { return { file: file, path: file.name }; });
        App.uploadItems(items);
        zipInput.value = '';
      });
    }

    if (!dropzone) { return; }
    ['dragenter', 'dragover'].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.add('dragging');
      });
    });
    ['dragleave', 'dragend'].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        dropzone.classList.remove('dragging');
      });
    });
    dropzone.addEventListener('drop', function (event) {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove('dragging');
      App.collectDrop(event.dataTransfer).then(function (items) {
        if (items.length) { App.uploadItems(items); }
      });
    });

    var register = App.$('#register-btn');
    var registerPath = App.$('#register-path');
    if (register && registerPath) {
      register.addEventListener('click', async function () {
        var path = registerPath.value.trim();
        if (!path) { return; }
        try {
          var result = await App.api('/api/datasets/register', {
            method: 'POST',
            body: JSON.stringify({ path: path })
          });
          registerPath.value = '';
          await App.loadDatasets(true);
          App.loadDatasetDetail(result.dataset.id);
          App.toast('数据集已注册');
        } catch (error) {
          App.toast('注册失败：' + error.message, 'error');
        }
      });
    }
  };

  App.collectDrop = function (dataTransfer) {
    if (!dataTransfer) { return Promise.resolve([]); }
    var entries = [];
    if (dataTransfer.items && dataTransfer.items.length) {
      for (var i = 0; i < dataTransfer.items.length; i += 1) {
        var entry = null;
        try {
          entry = dataTransfer.items[i].webkitGetAsEntry();
        } catch (error) {
          entry = null;
        }
        if (entry) { entries.push(entry); }
      }
    }
    if (!entries.length) {
      var files = Array.prototype.slice.call(dataTransfer.files || []);
      return Promise.resolve(files.map(function (file) {
        return { file: file, path: file.webkitRelativePath || file.name };
      }));
    }
    var output = [];
    return Promise.all(entries.map(function (entry) {
      return App.readEntry(entry, '', output);
    })).then(function () { return output; });
  };

  App.readEntry = function (entry, prefix, output) {
    return new Promise(function (resolve) {
      if (entry.isFile) {
        entry.file(function (file) {
          output.push({ file: file, path: prefix + file.name });
          resolve();
        }, function () { resolve(); });
        return;
      }
      if (entry.isDirectory) {
        var reader = entry.createReader();
        var children = [];
        var readBatch = function () {
          reader.readEntries(function (batch) {
            if (!batch.length) {
              Promise.all(children.map(function (child) {
                return App.readEntry(child, prefix + entry.name + '/', output);
              })).then(function () { resolve(); });
              return;
            }
            children = children.concat(Array.prototype.slice.call(batch));
            readBatch();
          }, function () { resolve(); });
        };
        readBatch();
        return;
      }
      resolve();
    });
  };

  App.uploadItems = function (items) {
    if (!items || !items.length) { return; }
    var isZip = items.length === 1 && /\.zip$/i.test(items[0].file.name || '');
    var nameInput = App.$('#upload-name');
    var name = nameInput ? nameInput.value.trim() : '';
    if (!name) {
      if (isZip) {
        name = items[0].file.name.replace(/\.zip$/i, '');
      } else {
        var first = items[0].path || '';
        name = first.split('/')[0] || 'upload_' + Date.now();
      }
    }
    var form = new FormData();
    items.forEach(function (item) { form.append('files', item.file, item.file.name); });
    if (!isZip) {
      form.append('paths', JSON.stringify(items.map(function (item) { return item.path; })));
    }
    form.append('name', name);

    var progress = App.$('#upload-progress');
    var bar = App.$('#upload-bar');
    var text = App.$('#upload-text');
    if (progress) { progress.classList.remove('hidden'); }
    if (bar) { bar.style.width = '0%'; }
    if (text) { text.textContent = '准备上传 ' + items.length + ' 个文件…'; }

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/datasets/upload');
    xhr.upload.onprogress = function (event) {
      if (!event.lengthComputable) { return; }
      var pct = Math.round((event.loaded / event.total) * 100);
      if (bar) { bar.style.width = pct + '%'; }
      if (text) { text.textContent = '上传中 ' + pct + '% · ' + (event.loaded / 1024 / 1024).toFixed(1) + ' MB'; }
    };
    xhr.onload = function () {
      if (progress) { progress.classList.add('hidden'); }
      var payload = {};
      try { payload = JSON.parse(xhr.responseText || '{}'); } catch (error) { payload = {}; }
      if (xhr.status >= 200 && xhr.status < 300 && payload.dataset) {
        if (nameInput) { nameInput.value = ''; }
        App.toast('数据集上传完成');
        App.loadDatasets(true).then(function () {
          App.loadDatasetDetail(payload.dataset.id);
        });
      } else {
        App.toast('上传失败：' + (payload.error || ('HTTP ' + xhr.status)), 'error');
      }
    };
    xhr.onerror = function () {
      if (progress) { progress.classList.add('hidden'); }
      App.toast('上传网络错误', 'error');
    };
    xhr.send(form);
  };
})();
