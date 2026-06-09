/**
 * 数字人播报系统 - 前端交互逻辑
 */

// 全局状态
const state = {
    avatar: null,
    background: null,
    voiceReference: null,
    tasks: [],
    pollingIntervals: {}
};

// DOM 元素引用
const elements = {
    // 上传输入
    avatarInput: document.getElementById('avatarInput'),
    bgInput: document.getElementById('bgInput'),
    audioInput: document.getElementById('audioInput'),
    
    // 上传区域
    avatarZone: document.getElementById('avatarZone'),
    bgZone: document.getElementById('bgZone'),
    audioZone: document.getElementById('audioZone'),
    
    // 预览
    avatarPreview: document.getElementById('avatarPreview'),
    bgPreview: document.getElementById('bgPreview'),
    audioPreview: document.getElementById('audioPreview'),
    
    // 占位符
    avatarPlaceholder: document.getElementById('avatarPlaceholder'),
    bgPlaceholder: document.getElementById('bgPlaceholder'),
    audioPlaceholder: document.getElementById('audioPlaceholder'),
    
    // 移除按钮
    removeAvatar: document.getElementById('removeAvatar'),
    removeBg: document.getElementById('removeBg'),
    removeAudio: document.getElementById('removeAudio'),
    
    // 配置
    voiceSelect: document.getElementById('voiceSelect'),
    speedRange: document.getElementById('speedRange'),
    speedValue: document.getElementById('speedValue'),
    textInput: document.getElementById('textInput'),
    charCount: document.getElementById('charCount'),
    generateBtn: document.getElementById('generateBtn'),
    
    // 任务
    tasksPanel: document.getElementById('tasksPanel'),
    tasksList: document.getElementById('tasksList'),
    
    // 弹窗
    resultModal: document.getElementById('resultModal'),
    resultVideo: document.getElementById('resultVideo'),
    downloadBtn: document.getElementById('downloadBtn'),
    modalClose: document.getElementById('modalClose'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    
    // 加载
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    
    // 提示
    toastContainer: document.getElementById('toastContainer'),

    promptText: document.getElementById('promptText')
};

// 初始化
function init() {
    bindEvents();
    loadVoices();
    loadTasks();
}

// 绑定事件
function bindEvents() {
    // 文件上传
    elements.avatarInput.addEventListener('change', (e) => handleFileSelect(e, 'avatar'));
    elements.bgInput.addEventListener('change', (e) => handleFileSelect(e, 'background'));
    elements.audioInput.addEventListener('change', (e) => handleFileSelect(e, 'voiceReference'));
    
    // 移除文件
    elements.removeAvatar.addEventListener('click', (e) => removeFile(e, 'avatar'));
    elements.removeBg.addEventListener('click', (e) => removeFile(e, 'background'));
    elements.removeAudio.addEventListener('click', (e) => removeFile(e, 'voiceReference'));
    
    // 配置
    elements.speedRange.addEventListener('input', updateSpeedDisplay);
    elements.textInput.addEventListener('input', updateCharCount);
    elements.generateBtn.addEventListener('click', generateVideo);
    
    // 弹窗
    elements.modalClose.addEventListener('click', closeModal);
    elements.closeModalBtn.addEventListener('click', closeModal);
    elements.resultModal.addEventListener('click', (e) => {
        if (e.target === elements.resultModal) closeModal();
    });
}

// 处理文件选择
async function handleFileSelect(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    showLoading('正在上传...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.code === 0) {
            state[type] = result.data.filename;
            showPreview(type, file, result.data.url);
            showToast('上传成功', 'success');
        } else {
            showToast(result.message || '上传失败', 'error');
        }
    } catch (error) {
        showToast('上传失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 显示预览
function showPreview(type, file, url) {
    if (type === 'avatar') {
        elements.avatarPreview.src = url;
        elements.avatarPreview.hidden = false;
        elements.avatarPlaceholder.hidden = true;
        elements.removeAvatar.hidden = false;
        elements.avatarZone.classList.add('has-file');
    } else if (type === 'background') {
        elements.bgPreview.src = url;
        elements.bgPreview.hidden = false;
        elements.bgPlaceholder.hidden = true;
        elements.removeBg.hidden = false;
        elements.bgZone.classList.add('has-file');
    } else if (type === 'voiceReference') {
        document.getElementById('audioName').textContent = file.name;
        elements.audioPreview.hidden = false;
        elements.audioPlaceholder.hidden = true;
        elements.removeAudio.hidden = false;
        elements.audioZone.classList.add('has-file');
    }
}

// 移除文件
function removeFile(event, type) {
    event.stopPropagation();
    state[type] = null;
    
    if (type === 'avatar') {
        elements.avatarInput.value = '';
        elements.avatarPreview.hidden = true;
        elements.avatarPreview.src = '';
        elements.avatarPlaceholder.hidden = false;
        elements.removeAvatar.hidden = true;
        elements.avatarZone.classList.remove('has-file');
    } else if (type === 'background') {
        elements.bgInput.value = '';
        elements.bgPreview.hidden = true;
        elements.bgPreview.src = '';
        elements.bgPlaceholder.hidden = false;
        elements.removeBg.hidden = true;
        elements.bgZone.classList.remove('has-file');
    } else if (type === 'voiceReference') {
        elements.audioInput.value = '';
        elements.audioPreview.hidden = true;
        elements.audioPlaceholder.hidden = false;
        elements.removeAudio.hidden = true;
        elements.audioZone.classList.remove('has-file');
    }
}

// 加载音色列表
async function loadVoices() {
    try {
        const response = await fetch('/api/voices');
        const result = await response.json();
        
        if (result.code === 0 && result.data) {
            elements.voiceSelect.innerHTML = result.data.map(v => 
                `<option value="${v.id}">${v.name} (${v.gender}) - ${v.style}</option>`
            ).join('');
        } else {
            elements.voiceSelect.innerHTML = '<option value="zh-CN-XiaoxiaoNeural">晓晓 (女)</option>';
        }
    } catch (error) {
        console.error('加载音色失败:', error);
        elements.voiceSelect.innerHTML = '<option value="zh-CN-XiaoxiaoNeural">晓晓 (女)</option>';
    }
}

// 更新语速显示
function updateSpeedDisplay() {
    elements.speedValue.textContent = elements.speedRange.value + 'x';
}

// 更新字数统计
function updateCharCount() {
    const count = elements.textInput.value.length;
    elements.charCount.textContent = `${count} / 5000`;
    
    if (count > 5000) {
        elements.charCount.style.color = '#ff4757';
    } else {
        elements.charCount.style.color = '#808090';
    }
}

// 生成视频
async function generateVideo() {
    // 验证
    if (!state.avatar) {
        showToast('请上传数字人形象照片', 'error');
        return;
    }
    
    const text = elements.textInput.value.trim();
    if (!text) {
        showToast('请输入播报文本', 'error');
        return;
    }
    
    const voice = elements.voiceSelect.value;
    const speed = parseFloat(elements.speedRange.value);
    
    showLoading('正在创建任务...');
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                avatar: state.avatar,
                background: state.background,
                voiceReference: state.voiceReference,
                voice: voice,
                text: text,
                speed: speed,
                promptText: elements.promptText.value.trim()
            })
        });
        
        const result = await response.json();
        
        if (result.code === 0) {
            showToast('任务创建成功，正在处理...', 'success');
            
            // 显示任务面板
            elements.tasksPanel.style.display = 'block';
            
            // 开始轮询任务状态
            startPolling(result.data.task_id);
            
            // 添加任务到列表
            addTaskToList({
                id: result.data.task_id,
                status: 'pending',
                progress: 0,
                message: '等待处理',
                text: text,
                created_at: new Date().toISOString()
            });
        } else {
            showToast(result.message || '创建任务失败', 'error');
        }
    } catch (error) {
        showToast('创建任务失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// 开始轮询任务状态
function startPolling(taskId) {
    if (state.pollingIntervals[taskId]) {
        clearInterval(state.pollingIntervals[taskId]);
    }
    
    state.pollingIntervals[taskId] = setInterval(async () => {
        try {
            const response = await fetch(`/api/task/${taskId}`);
            const result = await response.json();
            
            if (result.code === 0 && result.data) {
                updateTaskInList(result.data);
                
                // 任务完成或失败，停止轮询
                if (result.data.status === 'completed' || result.data.status === 'failed') {
                    clearInterval(state.pollingIntervals[taskId]);
                    delete state.pollingIntervals[taskId];
                    
                    if (result.data.status === 'completed') {
                        showToast('视频生成完成！', 'success');
                        showResult(result.data.result);
                    } else {
                        showToast('生成失败: ' + result.data.message, 'error');
                    }
                }
            }
        } catch (error) {
            console.error('轮询任务状态失败:', error);
        }
    }, 2000);
}

// 添加任务到列表
function addTaskToList(task) {
    state.tasks.unshift(task);
    renderTasks();
}

// 更新任务列表中的任务
function updateTaskInList(task) {
    const index = state.tasks.findIndex(t => t.id === task.id);
    if (index >= 0) {
        state.tasks[index] = { ...state.tasks[index], ...task };
    } else {
        state.tasks.unshift(task);
    }
    renderTasks();
}

// 渲染任务列表
function renderTasks() {
    if (state.tasks.length === 0) {
        elements.tasksPanel.style.display = 'none';
        return;
    }
    
    elements.tasksPanel.style.display = 'block';
    
    elements.tasksList.innerHTML = state.tasks.map(task => {
        const statusClass = `status-${task.status}`;
        const statusText = {
            'pending': '⏳ 等待中',
            'processing': '🔄 处理中',
            'completed': '✅ 已完成',
            'failed': '❌ 失败'
        }[task.status] || task.status;
        
        const isCompleted = task.status === 'completed';
        const hasResult = task.result && task.result.video_url;
        
        return `
            <div class="task-item" data-task-id="${task.id}">
                <div class="task-info">
                    <div class="task-text">${escapeHtml(task.text || '').substring(0, 50)}${(task.text || '').length > 50 ? '...' : ''}</div>
                    <div class="task-meta">
                        <span class="task-status ${statusClass}">${statusText}</span>
                        <span>${formatDate(task.created_at)}</span>
                    </div>
                    ${task.status === 'processing' || task.status === 'pending' ? `
                        <div class="task-progress">
                            <div class="task-progress-bar" style="width: ${task.progress || 0}%"></div>
                        </div>
                    ` : ''}
                    ${task.status === 'failed' ? `<div style="color: #ff4757; font-size: 0.8rem; margin-top: 5px;">${escapeHtml(task.message || '')}</div>` : ''}
                </div>
                <div class="task-actions">
                    ${isCompleted && hasResult ? `
                        <button class="btn-small btn-view" onclick="viewResult('${task.result.video_url}')">👁️ 预览</button>
                        <a href="${task.result.video_url}" class="btn-small btn-download" download>⬇️ 下载</a>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// 显示结果弹窗
function showResult(result) {
    if (!result || !result.video_url) return;
    
    elements.resultVideo.src = result.video_url;
    elements.downloadBtn.href = result.video_url;
    elements.downloadBtn.download = `digital_human_${Date.now()}.mp4`;
    elements.resultModal.classList.add('active');
}

// 查看结果
function viewResult(videoUrl) {
    elements.resultVideo.src = videoUrl;
    elements.downloadBtn.href = videoUrl;
    elements.downloadBtn.download = `digital_human_${Date.now()}.mp4`;
    elements.resultModal.classList.add('active');
}

// 关闭弹窗
function closeModal() {
    elements.resultModal.classList.remove('active');
    elements.resultVideo.pause();
    elements.resultVideo.src = '';
}

// 加载任务列表（页面刷新时）
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const result = await response.json();
        
        if (result.code === 0 && result.data) {
            state.tasks = result.data;
            renderTasks();
            
            // 对未完成的任务继续轮询
            result.data.forEach(task => {
                if (task.status === 'pending' || task.status === 'processing') {
                    startPolling(task.id);
                }
            });
        }
    } catch (error) {
        console.error('加载任务列表失败:', error);
    }
}

// 显示加载
function showLoading(text) {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

// 隐藏加载
function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

// 显示提示
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// 转义HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 格式化日期
function formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('zh-CN', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
